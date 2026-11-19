"""Agent wire schemas. Numeric constraints are also enforced locally."""
from copy import deepcopy
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

IDS = ("clip_openai_b32", "clip_openai_b16", "clip_laion_g14",
       "internvl35_projected", "gemma4_projected", "dinov3_vision")
NOISE_TYPES = Literal["perlin", "gabor", "phasor", "single_fourier", "turing", "worley", "wavelet",
    "gaussian", "poisson", "speckle", "uniform", "bernoulli", "beta", "binomial", "exponential",
    "cauchy", "student_t", "chi_square", "f_distribution", "gamma", "weibull", "laplace", "logistic"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Region(StrictModel):
    kind: Literal["bbox", "polygon", "global"]
    points: list[list[float]]

    @model_validator(mode="after")
    def valid(self):
        if any(len(p) != 2 or any(not 0 <= v <= 1 for v in p) for p in self.points):
            raise ValueError("Coordinates must be normalized [x,y] pairs")
        if self.kind == "global" and self.points:
            raise ValueError("Global extent has no points")
        if self.kind == "bbox" and (len(self.points) != 2 or any(self.points[0][i] >= self.points[1][i] for i in (0, 1))):
            raise ValueError("Invalid bbox")
        if self.kind == "polygon" and len(self.points) < 3:
            raise ValueError("Polygon needs at least three vertices")
        return self


class Point(StrictModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class Bounds(StrictModel):
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)


class Geometry(StrictModel):
    kind: Literal["bbox", "polygon"]
    bounds: Bounds | None
    vertices: list[Point] | None

    def canonical(self):
        if self.kind == "bbox":
            if self.bounds is None or self.vertices is not None:
                raise ValueError("bbox requires bounds and vertices=null")
            b = self.bounds
            return Region(kind="bbox", points=[[b.x_min, b.y_min], [b.x_max, b.y_max]])
        if self.bounds is not None or self.vertices is None:
            raise ValueError("polygon requires vertices and bounds=null")
        points = [(v.x, v.y) for v in self.vertices]
        if points and points[-1] == points[0]:
            points = points[:-1]
        if len(points) < 3 or len(set(points)) != len(points):
            raise ValueError("Polygon needs distinct ordered vertices")
        area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1]))
        if abs(area) < 1e-12:
            raise ValueError("Polygon has zero area")
        return Region(kind="polygon", points=[list(p) for p in points])


class Focus(StrictModel):
    region_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    geometry: Geometry
    clue: str


class GradientPlan(StrictModel):
    regions: list[Focus] = Field(min_length=1, max_length=12)
    instruction: str

    @model_validator(mode="after")
    def valid(self):
        if len({r.region_id for r in self.regions}) != len(self.regions):
            raise ValueError("Duplicate region IDs")
        for r in self.regions:
            r.geometry.canonical()
        return self


FaceAttribute = Literal['age', 'gender', 'race']


class AttributeClues(StrictModel):
    age: str | None
    gender: str | None
    race: str | None


class FaceFocus(StrictModel):
    region_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    geometry: Geometry
    target_attributes: list[FaceAttribute] = Field(min_length=1, max_length=3)
    clues: AttributeClues

    @model_validator(mode='after')
    def coherent_attributes(self):
        if len(set(self.target_attributes)) != len(self.target_attributes):
            raise ValueError('target_attributes must not contain duplicates')
        values = self.clues.model_dump()
        nonempty = {name for name, clue in values.items()
                    if clue is not None and clue.strip()}
        if nonempty != set(self.target_attributes):
            raise ValueError('Exactly targeted attributes require nonempty clues; all others must be null')
        self.geometry.canonical()
        return self


class FaceGradientPlan(StrictModel):
    regions: list[FaceFocus] = Field(min_length=1, max_length=12)
    instruction: str

    @model_validator(mode='after')
    def unique_regions(self):
        if len({region.region_id for region in self.regions}) != len(self.regions):
            raise ValueError('Duplicate region IDs')
        return self


class EncoderWeights(StrictModel):
    clip_openai_b32: float = Field(gt=0)
    clip_openai_b16: float = Field(gt=0)
    clip_laion_g14: float = Field(gt=0)
    internvl35_projected: float = Field(gt=0)
    gemma4_projected: float = Field(gt=0)
    dinov3_vision: float = Field(gt=0)


class Feather(StrictModel):
    region_id: str
    feather_pixels: float = Field(gt=0, le=64)


class GradientParameters(StrictModel):
    encoder_weights: EncoderWeights
    regions: list[Feather]
    seed: int = Field(ge=0, le=2147483647)
    explanation: str


class Parameter(StrictModel):
    name: str
    value: float


class NumericConfig(StrictModel):
    strength: float = Field(ge=0)
    color: list[float] = Field(min_length=3, max_length=3)
    seed: int = Field(ge=0, le=2147483647)
    feather_pixels: float = Field(gt=0, le=64)
    parameters: list[Parameter]

    @model_validator(mode="after")
    def unique(self):
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError("Duplicate noise parameters")
        return self


class ComponentIntent(StrictModel):
    component_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    action: Literal["keep", "add", "modify", "replace"]
    noise_type: NOISE_TYPES | None
    target_region: Geometry | None
    target_clue: str | None
    adjustment: str


class ToolPlan(StrictModel):
    parent_candidate_id: str | None
    components: list[ComponentIntent] = Field(min_length=1, max_length=32)
    instruction: str


class FaceComponentIntent(StrictModel):
    component_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    action: Literal["keep", "add", "modify", "replace"]
    noise_type: NOISE_TYPES | None
    target_region: Geometry | None
    target_attributes: list[FaceAttribute] | None
    clues: AttributeClues | None
    adjustment: str

    @model_validator(mode='after')
    def coherent_attributes(self):
        if self.action == 'keep':
            if any(value is not None for value in
                   (self.noise_type, self.target_region, self.target_attributes, self.clues)):
                raise ValueError('keep requires type, region, target_attributes, and clues to be null')
            return self
        if (self.noise_type is None or self.target_region is None or
                not self.target_attributes or self.clues is None):
            raise ValueError('Editable components require type, region, target_attributes, and clues')
        if len(set(self.target_attributes)) != len(self.target_attributes):
            raise ValueError('target_attributes must not contain duplicates')
        values = self.clues.model_dump()
        nonempty = {name for name, clue in values.items()
                    if clue is not None and clue.strip()}
        if nonempty != set(self.target_attributes):
            raise ValueError('Exactly targeted attributes require nonempty clues; all others must be null')
        self.target_region.canonical()
        return self


class FaceToolPlan(StrictModel):
    parent_candidate_id: str | None
    components: list[FaceComponentIntent] = Field(min_length=1, max_length=32)
    instruction: str


class ComponentConfig(StrictModel):
    component_id: str
    config: NumericConfig


class ToolParameters(StrictModel):
    components: list[ComponentConfig]
    explanation: str


class Location(StrictModel):
    status: Literal["predicted", "abstain"]
    name_en: str
    name_local: str

    @model_validator(mode="after")
    def coherent(self):
        if self.status == "abstain" and (self.name_en or self.name_local):
            raise ValueError("An abstention must have empty names")
        if self.status == "predicted" and not (self.name_en.strip() or self.name_local.strip()):
            raise ValueError("An empty prediction is not an abstention")
        return self


class Clue(StrictModel):
    kind: Literal["text", "architecture", "road", "vegetation", "other"]
    description: str
    region: Region
    supports: list[Literal["country", "province_or_state", "city"]]


class GeoPrediction(StrictModel):
    country: Location
    province_or_state: Location
    city: Location
    visual_clues: list[Clue]


class Evidence(StrictModel):
    region_id: str
    geometry: Geometry
    description: str

    @model_validator(mode='after')
    def valid_geometry(self):
        self.geometry.canonical()
        return self


class Answer(StrictModel):
    answer: str | None
    status: Literal["answered", "cannot_determine", "declined"]
    evidence: list[Evidence] = Field(max_length=12)

    @model_validator(mode="after")
    def coherent(self):
        if self.status == "answered" and not (self.answer or "").strip():
            raise ValueError("answered requires a nonempty answer")
        for evidence in self.evidence:
            evidence.geometry.canonical()
        return self


class FaceAttributePrediction(StrictModel):
    status: Literal['answered']
    evidence: list[Evidence] = Field(max_length=12)


class FaceAgePrediction(FaceAttributePrediction):
    answer: str

    @model_validator(mode='after')
    def valid_answer(self):
        answer = self.answer.strip().lower()
        if not (answer.isdigit() and 0 <= int(answer) <= 120):
            raise ValueError('Age answer must be one integer string from 0 to 120')
        self.answer = answer
        return self


class FaceGenderPrediction(FaceAttributePrediction):
    answer: Literal['female', 'male']


class FaceRacePrediction(FaceAttributePrediction):
    answer: Literal['white', 'black', 'east_asian', 'west_asian', 'east_asian/white']

    @model_validator(mode='after')
    def normalize_answer(self):
        answer = self.answer.strip().lower()
        self.answer = answer
        return self


class MedicalEvidence(StrictModel):
    target: Literal['age', 'sex']
    description: str
    bbox: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode='after')
    def valid_box(self):
        l, t, r, b = self.bbox
        if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
            raise ValueError('Medical evidence uses normalized left/top/right/bottom')
        return self


class MedicalMonthsPrediction(StrictModel):
    age_months: int = Field(ge=0, le=360)
    sex: Literal['male', 'female']
    evidence: list[MedicalEvidence] = Field(max_length=4)


class MedicalYearsPrediction(StrictModel):
    age: int = Field(ge=0, le=120)
    sex: Literal['male', 'female']
    evidence: list[MedicalEvidence] = Field(max_length=4)


class Artifact(StrictModel):
    region: Region
    severity: Literal["none", "minor", "moderate", "severe"]
    description: str


class Appearance(StrictModel):
    appearance_pass: bool
    artifacts: list[Artifact]
    explanation: str

    def passes(self):
        return self.appearance_pass and all(a.severity in ("none", "minor") for a in self.artifacts)


class FaceAppearance(StrictModel):
    overall_assessment: str
    damage_type: Literal['none', 'surface_variation', 'conspicuous_surface_contamination',
                         'strong_obstruction', 'dominant_discoloration',
                         'disruptive_bands_or_blocks', 'structural_distortion',
                         'multiple_overall_defects']
    appearance_pass: bool
    artifacts: list[Artifact]
    explanation: str

    @model_validator(mode='before')
    @classmethod
    def coordinate_format(cls, value):
        value = deepcopy(value)
        for artifact in value.get('artifacts', []):
            region = artifact.get('region', {})
            points = region.get('points', [])
            if (region.get('kind') == 'bbox' and points and
                    all(isinstance(point, list) and len(point) == 4 for point in points)):
                xs = [coordinate for point in points for coordinate in (point[0], point[2])]
                ys = [coordinate for point in points for coordinate in (point[1], point[3])]
                points = [[min(xs), min(ys)], [max(xs), max(ys)]]
            if points and all(len(point) == 2 for point in points):
                coordinates = [coordinate for point in points for coordinate in point]
                if (max(coordinates) > 1 and min(coordinates) >= 0 and
                        max(coordinates) <= 640 and all(float(v).is_integer() for v in coordinates)):
                    points = [[x/640, y/640] for x, y in points]
                if region.get('kind') == 'bbox':
                    xs = [point[0] for point in points]
                    ys = [point[1] for point in points]
                    points = [[min(xs), min(ys)], [max(xs), max(ys)]]
            region['points'] = points
        return value

    @model_validator(mode='after')
    def consistent_verdict(self):
        minor = all(artifact.severity in ('none', 'minor') for artifact in self.artifacts)
        intact = self.damage_type in ('none', 'surface_variation')
        if not (self.appearance_pass == minor == intact):
            raise ValueError('appearance_pass, damage_type, and artifact severities must agree')
        return self

    def passes(self):
        return self.appearance_pass and all(a.severity in ('none', 'minor') for a in self.artifacts)
