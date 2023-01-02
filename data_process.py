import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# import fire
import numpy as np
from datasets import load_dataset
from pydantic import BaseModel
from pydantic.main import Extra
from tqdm import tqdm
# from transformers.models.auto.tokenization_auto import AutoTokenizer

Span = Tuple[int, int]


class FlatQuintuplet(BaseModel):
    tokens: List[str]
    head: Span
    tail: Span
    value: Span
    relation: str
    qualifier: str


def load_quintuplets(path: str) -> List[FlatQuintuplet]:
    with open(path) as f:
        return [FlatQuintuplet(**json.loads(line)) for line in f]


class Entity(BaseModel):
    span: Span  # Token spans, start inclusive, end exclusive
    label: str

    def as_tuple(self) -> Tuple[int, int, str]:
        return self.span[0], self.span[1], self.label


class Relation(BaseModel):
    head: Span
    tail: Span
    label: str
    qualifiers: List[Entity]

    def merge(self, other):
        assert isinstance(other, Relation)
        assert (self.head, self.tail, self.label) == (
            other.head,
            other.tail,
            other.label,
        )
        qualifiers: Dict[str, Entity] = {q.json(): q for q in self.qualifiers}
        for q in other.qualifiers:
            qualifiers[q.json()] = q
        self.qualifiers = list(qualifiers.values())

    def as_tuples(self, tokens: List[str]) -> List[Tuple[str, str, str, str, str]]:
        tuples = []
        head = " ".join(tokens[slice(*self.head)])
        tail = " ".join(tokens[slice(*self.tail)])
        for q in self.qualifiers:
            value = " ".join(tokens[slice(*q.span)])
            tuples.append((head, self.label, tail, q.label, value))
        return tuples


class SparseCube(BaseModel):
    shape: Tuple[int, int, int]
    entries: List[Tuple[int, int, int, int]]

    def check_equal(self, other):
        assert isinstance(other, SparseCube)
        return self.shape == other.shape and set(self.entries) == set(other.entries)

    @classmethod
    def from_numpy(cls, x: np.ndarray):
        entries = []
        i_list, j_list, k_list = x.nonzero()
        for i, j, k in zip(i_list, j_list, k_list):
            entries.append((i, j, k, x[i, j, k]))
        return cls(shape=tuple(x.shape), entries=entries)

    def numpy(self) -> np.ndarray:
        x = np.zeros(shape=self.shape)
        for i, j, k, value in self.entries:
            x[i, j, k] = value
        return x

    def tolist(self) -> List[List[List[int]]]:
        x = self.numpy()
        return [[list(row) for row in table] for table in x]

    def numel(self) -> int:
        i, j, k = self.shape
        return i * j * k

    @classmethod
    def empty(cls):
        return cls(shape=(0, 0, 0), entries=[])


class Sentence(BaseModel):
    tokens: List[str]
    entities: List[Entity]
    relations: List[Relation]
    wordpieceSentText: Optional[str]
    wordpieceTokensIndex: Optional[List[Span]]
    wordpieceSegmentIds: Optional[List[int]]
    jointLabelMatrix: Optional[List[List[int]]]
    quintupletMatrix: Optional[SparseCube]

    def check_span_overlap(self) -> bool:
        entity_pos = [0 for _ in range(9999)]
        for e in self.entities:
            st, ed = e.span
            for i in range(st, ed):
                if entity_pos[i] != 0:
                    return True
                entity_pos[i] = 1
        return False

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def merge(self, other):
        if other is None:
            return

        assert isinstance(other, Sentence)
        assert other.text == self.text

        ents = {e.json(): e for e in self.entities}
        for e in other.entities:
            ents[e.json()] = e
        self.entities = list(ents.values())

        relations = {(r.head, r.tail, r.label): r for r in self.relations}
        for r in other.relations:
            key = (r.head, r.tail, r.label)
            if key not in relations.keys():
                relations[key] = r
            else:
                relations[key].merge(r)
            assert relations[key] is not None

        self.relations = list(relations.values())


class Data(BaseModel):
    sents: List[Sentence]

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            lines = f.readlines()

        sents = [Sentence(**json.loads(line)) for line in tqdm(lines, desc=path)]
        return cls(sents=sents)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.sents:
                raw = s.dict()
                raw = {k: v for k, v in raw.items() if v is not None}
                f.write(json.dumps(raw) + "\n")

    def to_flat_quintuplets(self) -> List[FlatQuintuplet]:
        outputs = []
        for s in tqdm(self.sents, desc="to_flat_quintuplets"):
            for r in s.relations:
                for q in r.qualifiers:
                    flat = FlatQuintuplet(
                        tokens=s.tokens,
                        head=r.head,
                        tail=r.tail,
                        relation=r.label,
                        qualifier=q.label,
                        value=q.span,
                    )
                    outputs.append(flat)
        return outputs

    @classmethod
    def load_from_flat_quintuplets(cls, path: str):
        quintuplets = load_quintuplets(path)
        mapping: Dict[str, Sentence] = {}

        for q in tqdm(quintuplets, desc="load_from_flat_quintuplets"):
            ents = [
                Entity(span=span, label="Entity") for span in [q.head, q.tail, q.value]
            ]
            relation = Relation(
                head=q.head,
                tail=q.tail,
                label=q.relation,
                qualifiers=[Entity(span=q.value, label=q.qualifier)],
            )
            sent = Sentence(tokens=q.tokens, entities=ents, relations=[relation])
            sent.merge(mapping.get(sent.text))
            assert sent is not None
            mapping[sent.text] = sent

        data = cls(sents=list(mapping.values()))
        old = set(flat.json() for flat in quintuplets)
        new = set(flat.json() for flat in data.to_flat_quintuplets())
        assert old == new
        return data

    def analyze(self):
        relation_labels = []
        qualifier_labels = []
        for s in self.sents:
            for r in s.relations:
                relation_labels.append(r.label)
                for q in r.qualifiers:
                    qualifier_labels.append(q.label)

        info = dict(
            sents=len(self.sents),
            relations=len(relation_labels),
            relation_labels=len(set(relation_labels)),
            qualifiers=len(qualifier_labels),
            qualifier_labels=len(set(qualifier_labels)),
            hash=hashlib.md5(self.json().encode()).hexdigest(),
        )
        print(json.dumps(info, indent=2))


class RawPred(BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    tokens: np.ndarray
    joint_label_matrix: np.ndarray
    joint_label_preds: np.ndarray
    quintuplet_preds: SparseCube = SparseCube.empty()
    all_separate_position_preds: List[int]
    all_ent_preds: Dict[Span, str]
    all_rel_preds: Dict[Tuple[Span, Span], str]
    all_q_preds: Dict[Tuple[Span, Span, Span], str] = {}
    all_rel_probs: Dict[Tuple[Span, Span], float] = {}
    all_q_probs: Dict[Tuple[Span, Span, Span], float] = {}

    def assert_valid(self):
        assert self.tokens.size > 0
        assert self.joint_label_matrix.size > 0
        assert self.joint_label_preds.size > 0

    @classmethod
    def empty(cls):
        return cls(
            tokens=np.array([]),
            joint_label_matrix=np.empty(shape=(1,)),
            joint_label_preds=np.empty(shape=(1,)),
            all_separate_position_preds=[],
            all_ent_preds={},
            all_rel_preds={},
        )

    def check_if_empty(self):
        return len(self.tokens) == 0

    def has_relations(self) -> bool:
        return len(self.all_rel_preds.keys()) > 0

    def as_sentence(self, vocab) -> Sentence:
        tokens = [vocab.get_token_from_index(i, "tokens") for i in self.tokens]
        tokens = [t for t in tokens if t != vocab.DEFAULT_PAD_TOKEN]

        span_to_ent = {}
        for span, label in self.all_ent_preds.items():
            e = Entity(span=span, label=label)
            span_to_ent[span] = e

        pair_to_relation = {}
        for (head, tail), label in self.all_rel_preds.items():
            r = Relation(head=head, tail=tail, label=label, qualifiers=[])
            pair_to_relation[(head, tail)] = r

        for (head, tail, value), label in self.all_q_preds.items():
            q = Entity(span=value, label=label)
            pair_to_relation[(head, tail)].qualifiers.append(q)

        return Sentence(
            tokens=tokens,
            entities=list(span_to_ent.values()),
            relations=list(pair_to_relation.values()),
        )


def add_tokens(sent, tokenizer):
    cls = tokenizer.cls_token
    sep = tokenizer.sep_token
    wordpiece_tokens = [cls, sep]
    is_roberta = "roberta" in type(tokenizer).__name__.lower()
    if is_roberta:
        wordpiece_tokens.pop()  # RoBERTa format is [cls, tokens, sep, pad]

    context_len = len(wordpiece_tokens)
    wordpiece_segment_ids = [0] * context_len

    wordpiece_tokens_index = []
    cur_index = len(wordpiece_tokens)
    for token in sent["tokens"]:
        if is_roberta:
            token = " " + token  # RoBERTa is space-sensitive
        tokenized_token = list(tokenizer.tokenize(token))
        wordpiece_tokens.extend(tokenized_token)
        wordpiece_tokens_index.append([cur_index, cur_index + len(tokenized_token)])
        cur_index += len(tokenized_token)
    wordpiece_tokens.append(sep)
    wordpiece_segment_ids += [1] * (len(wordpiece_tokens) - context_len)

    sent.update(
        {
            "wordpieceSentText": " ".join(wordpiece_tokens),
            "wordpieceTokensIndex": wordpiece_tokens_index,
            "wordpieceSegmentIds": wordpiece_segment_ids,
        }
    )
    return sent


def add_joint_label(sent, label_vocab):
    """add_joint_label add joint labels for sentences"""

    ent_rel_id = label_vocab["id"]
    none_id = ent_rel_id["None"]
    seq_len = len(sent["tokens"])
    label_matrix = [[none_id for _ in range(seq_len)] for _ in range(seq_len)]

    for ent in sent["entities"]:
        for i in range(ent["span"][0], ent["span"][1]):
            for j in range(ent["span"][0], ent["span"][1]):
                label_matrix[i][j] = ent_rel_id[ent["label"]]

    entries: List[Tuple[int, int, int, int]] = []
    for rel in sent["relations"]:
        for i in range(rel["head"][0], rel["head"][1]):
            for j in range(rel["tail"][0], rel["tail"][1]):
                label_matrix[i][j] = ent_rel_id[rel["label"]]
                for q in rel["qualifiers"]:
                    for k in range(q["span"][0], q["span"][1]):
                        entries.append((i, j, k, ent_rel_id[q["label"]]))

    sent["jointLabelMatrix"] = label_matrix
    sent["quintupletMatrix"] = SparseCube(
        shape=(seq_len, seq_len, seq_len), entries=entries
    ).dict()
    return sent


def add_tag_joint_label(sent, label_vocab):
    ent_rel_id = label_vocab["id"]
    none_id = ent_rel_id["O"]
    seq_len = len(sent["tokens"])
    label_matrix = [[none_id for _ in range(seq_len)] for _ in range(seq_len)]

    spans = [Entity(**e).as_tuple() for e in sent["entities"]]
    encoder = BioEncoder()
    tags = encoder.run(spans, seq_len)
    if not sorted(encoder.decode(tags)) == sorted(spans):
        print(dict(gold=sorted(spans), decoded=sorted(encoder.decode(tags))))

    assert len(tags) == seq_len
    for i, t in enumerate(tags):
        label_matrix[i][i] = ent_rel_id[t]  # We only care about diagonal here

    sent["jointLabelMatrix"] = label_matrix
    sent["quintupletMatrix"] = SparseCube.empty().dict()
    return sent


def process(
    source_file: str,
    target_file: str,
    label_file: str = "data/quintuplet/label_vocab.json",
    pretrained_model: str = "bert-base-uncased",
    mode: str = "",
):
    print(dict(process=locals()))
    #auto_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    #print("Load {} tokenizer successfully.".format(pretrained_model))

    with open(label_file) as f:
        label_vocab = json.load(f)

    with open(source_file) as fin, open(target_file, "w") as fout:
        for line in tqdm(fin.readlines()):
            if mode == "tags":
                s = Sentence(**json.loads(line))
                for s in convert_sent_to_tags(s):
                    sent = s.dict()
                    sent = add_tokens(sent, auto_tokenizer)
                    sent = add_tag_joint_label(sent, label_vocab)
                    print(json.dumps(sent), file=fout)
            else:
                sent = json.loads(line.strip())
                processed = dict()
                processed['sentences'] = [sent['tokens']]
                processed_ner=[]
                for entitie in sent['entities']:
                    processed_ner.append([entitie['span'][0], entitie['span'][1]-1, entitie['label']])
                processed['ner'] = [processed_ner]
                processed_relations = []
                for relation in sent['relations']:
                    processed_q = []
                    for q in relation['qualifiers']:
                        processed_q.append([q['span'][0], q['span'][1]-1, q['label']])
                    processed_relations.append([relation['head'][0], relation['head'][1]-1, relation['tail'][0], relation['tail'][1]-1, relation['label'], processed_q])
                processed['relations'] = [processed_relations]
                processed['clusters'] = []
                processed['doc_key'] = ""
                print(json.dumps(processed), file=fout)


def make_label_file(pattern_in: str, path_out: str):
    sents = []
    for path in sorted(Path().glob(pattern_in)):
        with open(path) as f:
            sents.extend([Sentence(**json.loads(line)) for line in tqdm(f)])

    relations = sorted(set(r.label for s in sents for r in s.relations))
    qualifiers = sorted(
        set(q.label for s in sents for r in s.relations for q in r.qualifiers)
    )
    labels = ["None", "Entity"] + qualifiers + sorted(set(relations) - set(qualifiers))
    label_map = {name: i for i, name in enumerate(labels)}
    print(dict(relations=len(relations), qualifiers=len(qualifiers)))

    info = dict(
        id=label_map,
        symmetric=[],
        asymmetric=[],
        entity=[label_map["Entity"]],
        relation=[label_map[name] for name in relations],
        qualifier=[label_map[name] for name in qualifiers],
        q_num_logits=len(qualifiers) + 2,
    )
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    with open(path_out, "w") as f:
        f.write(json.dumps(info, indent=2))


def make_tag_label_file(pattern_in: str, path_out: str):
    tags = []
    qualifiers = []
    for path in sorted(Path().glob(pattern_in)):
        with open(path) as f:
            for line in tqdm(f):
                s = Sentence(**json.loads(line))
                for q in [q for r in s.relations for q in r.qualifiers]:
                    tags.append("B-" + q.label)
                    tags.append("I-" + q.label)
                    qualifiers.append(q.label)  # Dataset reader needs it

    tags = sorted(set(tags))
    qualifiers = sorted(set(qualifiers))
    labels = ["O"] + tags + qualifiers
    info = dict(
        id={name: i for i, name in enumerate(labels)},
        q_num_logits=len(tags) + 1,
    )
    print(dict(labels=len(labels), tags=len(tags), qualifiers=len(qualifiers)))
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    with open(path_out, "w") as f:
        f.write(json.dumps(info, indent=2))


def convert_sent_to_tags(sent: Sentence) -> List[Sentence]:
    outputs = []
    for r in sent.relations:
        head = " ".join(sent.tokens[slice(*r.head)])
        tail = " ".join(sent.tokens[slice(*r.tail)])
        parts = [sent.text, head, r.label, tail]
        text = " | ".join(parts)

        new = sent.copy(deep=True)
        new.tokens = text.split()
        new.entities = r.qualifiers
        new.relations = []
        outputs.append(new)

    return outputs


def load_raw_preds(path: str) -> List[RawPred]:
    raw_preds = []
    with open(path, "rb") as f:
        raw = pickle.load(f)
        for r in raw:
            # noinspection Pydantic
            p = RawPred(**r)
            p.assert_valid()
            raw_preds.append(p)
    return raw_preds


def process_many(
    dir_in: str,
    dir_out: str,
    dir_temp: str = "temp",
    mode: str = "joint",
    **kwargs,
):
    if Path(dir_temp).exists():
        shutil.rmtree(dir_temp)
    for path in sorted(Path(dir_in).glob("*.json")):
        data = Data.load(str(path))
        data.analyze()
        data.save(str(Path(dir_temp) / path.name))

    path_label = str(Path(dir_out) / "label.json")
    if mode == "tags":
        make_tag_label_file(f"{dir_temp}/*.json", path_label)
    else:
        make_label_file(f"{dir_temp}/*.json", path_label) # label.json
    for path in sorted(Path(dir_temp).glob("*.json")):
        process(
            str(path), str(Path(dir_out) / path.name), path_label, mode=mode, **kwargs # train,dev,test
        )
    shutil.rmtree(dir_temp)


class BioEncoder:
    def run(self, spans: List[Tuple[int, int, str]], length: int) -> List[str]:
        assert self is not None
        tags = ["O" for _ in range(length)]
        for start, end, label in spans:
            assert start < end
            assert end <= length
            for i in range(start, end):
                tags[i] = "I-" + label
            tags[start] = "B-" + label
        return tags

    def decode(self, tags: List[str]) -> List[Tuple[int, int, str]]:
        assert self is not None
        parts = []
        for i, t in enumerate(tags):
            assert t[0] in "BIO"
            if t.startswith("B"):
                parts.append([i])
            if parts and t.startswith("I"):
                parts[-1].append(i)

        spans = []
        for indices in parts:
            if indices:
                start = min(indices)
                end = max(indices) + 1
                label = tags[start].split("-", maxsplit=1)[1]
                spans.append((start, end, label))

        return spans


def test_bio():
    encoder = BioEncoder()
    spans = [(0, 3, "one"), (3, 4, "one"), (7, 8, "three")]
    tags = encoder.run(spans, 8)
    preds = encoder.decode(tags)
    print(dict(spans=spans))
    print(dict(tags=tags))
    print(dict(pred=preds))
    assert spans == preds


def test_data(path: str):
    data = Data.load(path)
    data.analyze()

    for s in data.sents[:3]:
        print(f"\nText: {s.text}")
        print(f"Tokens: {s.tokens}")
        for r in s.relations:
            fn = lambda span: " ".join(s.tokens[span[0] : span[1]])
            print(f"\tRelation: {r}")
            print(f"\tHead: {fn(r.head)}, Relation: {r.label}, Tail: {fn(r.tail)}")
            for q in r.qualifiers:
                print(f"\t\tQualifier: {q.label}, Value: {fn(q.span)}")
        print()


def convert_flat(path_in: str, path_out: str):
    data = Data.load_from_flat_quintuplets(path_in)
    data.analyze()
    data.save(path_out)


def download_data(folder_out: str, name: str = "declare-lab/HyperRED"):
    dataset = load_dataset(name)
    for key, name in dict(train="train", validation="dev", test="test").items():
        data = Data(sents=[Sentence(**raw) for raw in dataset[key]])
        path_out = Path(folder_out, name).with_suffix(".json")
        data.save(str(path_out))
        print(dict(path_out=path_out))


"""
p data_process.py convert_flat data/flat_min_10/dev.json data/hyperred/dev.json
p data_process.py convert_flat data/flat_min_10/test.json data/hyperred/test.json
p data_process.py convert_flat data/flat_min_10/train.json data/hyperred/train.json

p data_process.py download_data data/hyperred/
p data_process.py process_many data/hyperred/ data/processed/
p data_process.py process_many data/hyperred/ data/processed_tags/ --mode tags

"""


if __name__ == "__main__":
    #fire.Fire()
    download_data("datasets/hyperred-raw/")
    process_many("datasets/hyperred-raw/","datasets/hyperred/")
    
    # from data_process import Data

    # path = "data/hyperred/train.json"
    # data = Data.load(path)

    # for s in data.sents[:3]:
    #     print()
    #     print(s.tokens)
    #     for r in s.relations:
    #         print(r.head, r.label, r.tail)
    #         for q in r.qualifiers:
    #             print(q.label, q.span)
