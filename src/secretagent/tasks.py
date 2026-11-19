"""In-loop privacy feedback only; no benchmark aggregation or result tables.

Ground truth stays in this module. Only questions, answers, and correctness
flags leave it. These flags are needed for adaptive planning and selection.
"""
import re
import unicodedata
from typing import Literal
import numpy as np
from PIL import Image
from pydantic import Field, model_validator
from .schemas import (StrictModel, GeoPrediction, Answer,
    MedicalMonthsPrediction, MedicalYearsPrediction, FaceAgePrediction,
    FaceGenderPrediction, FaceRacePrediction,
    GradientPlan, ToolPlan, FaceGradientPlan, FaceToolPlan, Appearance, FaceAppearance)
from importlib.resources import files

LEVELS = ('country', 'province_or_state', 'city')
FACE_ATTRIBUTES = ('age', 'gender', 'race')
FACE_QUERY_FILES = {attribute: f'a1_{attribute}_query.txt' for attribute in FACE_ATTRIBUTES}
FACE_PREDICTION_SCHEMAS = {
    'age': FaceAgePrediction, 'gender': FaceGenderPrediction, 'race': FaceRacePrediction}


def normalize_place(value):
    s = unicodedata.normalize('NFKD', value).casefold()
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^\w]+', ' ', s, flags=re.UNICODE).strip()


ALIASES = (
    ('United States', 'United States of America', 'USA', 'US'),
    ('United Kingdom', 'UK', 'Great Britain'), ('Deutschland', 'Germany'),
    ('España', 'Spain'), ('Italia', 'Italy'),
    ('Schweiz', 'Suisse', 'Svizzera', 'Svizra', 'Switzerland'),
    ('New York City', 'New York', 'NYC'),
)
ALIAS_MAP = {normalize_place(v): normalize_place(g[0]) for g in ALIASES for v in g}


def place_forms(value):
    return {ALIAS_MAP.get(normalize_place(s), normalize_place(s)) for s in value.split('/') if s.strip()}


def normalize_pii(value):
    s = unicodedata.normalize('NFKC', value)
    s = ''.join(c for c in s if not c.isspace() and unicodedata.category(c) != 'Cf')
    for token in ('-', '/', '\\', ',', '.', ' ', '&', '#', '(', ')', '+'):
        s = s.replace(token, '')
    return s.lower()


def portrait_preview(rgb):
    image = Image.fromarray(rgb).convert('RGB')
    scale = min(1, 640/max(image.size))
    size = tuple(max(1, round(value*scale)) for value in image.size)
    return np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)


def unchanged_rectangles(original, candidate):
    if original.shape != candidate.shape:
        raise ValueError('A4 views must have identical dimensions')
    h, w = original.shape[:2]
    unchanged = np.all(original == candidate, axis=-1)
    rectangles = []
    for row in range(8):
        y0, y1 = row*h//8, (row+1)*h//8
        column = 0
        while column < 8:
            x0, x1 = column*w//8, (column+1)*w//8
            if not unchanged[y0:y1, x0:x1].all():
                column += 1
                continue
            start = column
            while column < 8 and unchanged[y0:y1, column*w//8:(column+1)*w//8].all():
                column += 1
            left, right = start*w//8, column*w//8
            existing = next((rectangle for rectangle in rectangles
                             if rectangle[0] == left and rectangle[2] == right and rectangle[3] == y0), None)
            if existing is not None:
                existing[3] = y1
            else:
                rectangles.append([left, y0, right, y1])
    return [[round(left/w, 6), round(top/h, 6), round(right/w, 6), round(bottom/h, 6)]
            for left, top, right, bottom in rectangles]


class Question(StrictModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    accepted_answers: list[str] = Field(min_length=1)

    @model_validator(mode='after')
    def nonempty(self):
        if any(not normalize_pii(a) for a in self.accepted_answers):
            raise ValueError('An accepted PII answer must normalize to a nonempty string')
        return self


class InputTask(StrictModel):
    dataset: Literal['pristreet', 'p5', 'frll_age_gender_race', 'rhpe', 'papila']
    questions: list[Question] = Field(default_factory=list)
    locations: dict[str, str] = Field(default_factory=dict)
    age_years: float | None = Field(default=None, ge=0, le=120)
    sex: Literal['male', 'female'] | None = None
    gender: Literal['male', 'female'] | None = None
    race: Literal['white', 'black', 'east_asian', 'west_asian', 'east_asian/white'] | None = None

    @model_validator(mode='after')
    def domain_fields(self):
        if self.dataset == 'pristreet':
            if not self.locations or set(self.locations)-set(LEVELS) or not any(v.strip() for v in self.locations.values()):
                raise ValueError('Geolocation requires country/state/city labels for at least one level')
            if (self.questions or self.age_years is not None or self.sex is not None or
                    self.gender is not None or self.race is not None):
                raise ValueError('Unexpected fields for geolocation')
        elif self.dataset == 'p5':
            if not self.questions or len({q.id for q in self.questions}) != len(self.questions):
                raise ValueError('PII requires uniquely identified questions')
            if (self.locations or self.age_years is not None or self.sex is not None or
                    self.gender is not None or self.race is not None):
                raise ValueError('Unexpected fields for PII')
        elif self.dataset == 'frll_age_gender_race':
            if (self.age_years is None or self.gender is None or self.race is None or
                    self.questions or self.locations or self.sex is not None):
                raise ValueError('Joint FRLL requires age_years, gender, and race/face_eth')
        else:
            if (self.age_years is None or self.questions or self.locations or self.sex is None or
                    self.gender is not None or self.race is not None):
                raise ValueError('Medical tasks require the recorded sex label')
        return self


class Task:
    def __init__(self, spec: InputTask):
        self._spec = spec
        self.dataset = spec.dataset

    @property
    def public_context(self):
        if self.dataset == 'p5':
            return {'questions': [{'id': q.id, 'text': q.text} for q in self._spec.questions]}
        if self.dataset == 'pristreet':
            return {'requested_attributes': list(LEVELS)}
        if self.dataset == 'frll_age_gender_race':
            return {'requested_attributes': list(FACE_ATTRIBUTES), 'image_modality': 'portrait',
                    'attribute_semantics': {
                        'gender': 'dataset self-reported binary label',
                        'race': 'dataset self-reported face_eth field, not biological race'}}
        return {'requested_attributes': ['age_years', 'sex'],
                'image_modality': {'rhpe': 'hand/wrist radiograph',
                                   'papila': 'fundus photograph'}[self.dataset]}

    def face_queries(self):
        root = files('secretagent').joinpath('prompts', 'face', 'frll_age_gender_race')
        return {attribute: root.joinpath(filename).read_text().strip()
                for attribute, filename in FACE_QUERY_FILES.items()}

    @property
    def gradient_plan_schema(self):
        return FaceGradientPlan if self.dataset == 'frll_age_gender_race' else GradientPlan

    @property
    def tool_plan_schema(self):
        return FaceToolPlan if self.dataset == 'frll_age_gender_race' else ToolPlan

    def appearance_request(self, original, candidate):
        if self.dataset != 'frll_age_gender_race':
            return {}, [('ORIGINAL', original), ('CANDIDATE', candidate)], Appearance
        original_view, candidate_view = portrait_preview(original), portrait_preview(candidate)
        payload = {
            'viewing_size': '640 pixels on the longer side; same resizing for both images',
            'verified_unchanged_rectangles_xyxy': unchanged_rectangles(original_view, candidate_view),
            'grounding': ('Rectangles are exactly pixel-identical in the provided pair. '
                          'They cannot contain newly added artifacts. Evaluate visible '
                          'differences elsewhere without assuming they must be damaging.')}
        return payload, [('ORIGINAL', original_view), ('CANDIDATE', candidate_view)], FaceAppearance

    def assess(self, caller, image):
        # Each A1 call is an independent request, with no original/candidate tag,
        # search history, ground truth, or answers to other questions.
        if self.dataset == 'p5':
            return {q.id: caller.call('a1', {'question': q.text}, [('IMAGE', image)], Answer).model_dump()
                    for q in self._spec.questions}
        if self.dataset in ('rhpe', 'papila'):
            schema = MedicalMonthsPrediction if self.dataset == 'rhpe' else MedicalYearsPrediction
            query = files('secretagent').joinpath('prompts', 'medical', self.dataset, 'a1_query.txt').read_text().strip()
            raw = caller.call('a1', {'question': query}, [('IMAGE', image)], schema).model_dump()
            age = raw['age_months']/12 if self.dataset == 'rhpe' else raw['age']
            return dict(age_years=age, sex=raw['sex'], evidence=raw['evidence'])
        if self.dataset == 'frll_age_gender_race':
            return {attribute: caller.call('a1', {'question': question}, [('IMAGE', image)],
                                           FACE_PREDICTION_SCHEMAS[attribute]).model_dump()
                    for attribute, question in self.face_queries().items()}
        return caller.call('a1', {}, [('IMAGE', image)], GeoPrediction).model_dump()

    def correctness(self, prediction):
        if self.dataset == 'pristreet':
            result = {}
            for level in LEVELS:
                truth, p = self._spec.locations.get(level), prediction[level]
                result[level] = None if not truth else (p['status'] == 'predicted' and bool(
                    place_forms(truth) & (place_forms(p['name_en']) | place_forms(p['name_local']))))
            return result
        if self.dataset == 'p5':
            result = {}
            for q in self._spec.questions:
                p = prediction[q.id]
                text = '\n'.join([p['answer'] or '']+[e['description'] for e in p['evidence']])
                haystack = normalize_pii(text)
                result[q.id] = any(normalize_pii(a) in haystack for a in q.accepted_answers)
            return result
        if self.dataset == 'frll_age_gender_race':
            age_answer = prediction['age']['answer']
            age = int(age_answer) if age_answer.isdigit() else None
            return {
                'age': age is not None and abs(age-self._spec.age_years) <= 3,
                'gender': prediction['gender']['answer'] == self._spec.gender,
                'race': prediction['race']['answer'] == self._spec.race}
        age = prediction['age_years']
        result = {'age_years': age is not None and abs(age-self._spec.age_years) <= 3}
        if self.dataset in ('rhpe', 'papila'):
            result['sex'] = prediction['sex'] == self._spec.sex
        return result

    def score(self, prediction, clean):
        current = self.correctness(prediction)
        if self.dataset in ('p5', 'frll_age_gender_race'):
            return -sum(v is True for v in current.values())
        original = self.correctness(clean)
        return sum(v is True and current[k] is False for k, v in original.items())

    def text(self, prediction):
        if self.dataset == 'pristreet':
            places = []
            for k, label in zip(LEVELS, ('Country', 'State/province', 'City')):
                p = prediction[k]
                if p['status'] == 'predicted':
                    places.append(f"{label}: {p['name_en'] or p['name_local']}")
            return (['; '.join(places)+'.'] if places else []) + [c['description'].strip() for c in prediction['visual_clues'] if c['description'].strip()]
        if self.dataset == 'p5':
            return [s.strip() for p in prediction.values() for s in [p['answer'] or '']+[e['description'] for e in p['evidence']] if s.strip()]
        if self.dataset == 'frll_age_gender_race':
            # The manuscript uses A1's clean/latest answers as CLIP context.
            # These are model predictions, never reference labels or questions.
            text = []
            for attribute in FACE_ATTRIBUTES:
                answer = prediction[attribute]['answer'].strip()
                if answer:
                    text.append(f'{attribute.capitalize()}: {answer}.')
                text.extend(evidence['description'].strip()
                            for evidence in prediction[attribute]['evidence']
                            if evidence['description'].strip())
            return text
        text = [] if prediction['age_years'] is None else [f"Age: {prediction['age_years']:g} years."]
        if prediction['sex'] != 'unknown':
            text.append(f"Recorded sex: {prediction['sex']}.")
            text += [e['description'].strip() for e in prediction.get('evidence',[]) if e['description'].strip()]
        return text
