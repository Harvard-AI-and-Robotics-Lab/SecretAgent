"""Thirteen independent additive RGB distribution fields, version 1.

Finite-variance families use analytical centering/scaling, never sample-wise
normalization. Cauchy has a location/scale definition, not a mean/std definition.
These are image-noise adaptations of distribution families, not a raw random-number sampling task. Masking/projection/quantization happen later.
"""
from __future__ import annotations

import math

import numpy as np


STD = (0.05, 2, 0.5, False)
DISTRIBUTION_DOMAINS = {
    "uniform": {"std": STD},
    "bernoulli": {"p": (0.01, 0.99, 0.7, False), "std": STD},
    "beta": {"alpha": (0.1, 20, 2, False), "beta": (0.1, 20, 2, False), "std": STD},
    "binomial": {"n": (1, 1000, 10, True), "p": (0.01, 0.99, 0.5, False), "std": STD},
    "exponential": {"std": STD},
    "cauchy": {"scale": (0.01, 2, 0.5, False)},
    "student_t": {"df": (2.1, 100, 3, False), "std": STD},
    "chi_square": {"df": (0.5, 100, 5, False), "std": STD},
    "f_distribution": {"dfnum": (0.5, 100, 5, False), "dfden": (4.1, 100, 10, False), "std": STD},
    "gamma": {"shape": (0.1, 20, 2, False), "std": STD},
    "weibull": {"shape": (0.5, 10, 1.5, False), "std": STD},
    "laplace": {"std": STD},
    "logistic": {"std": STD},
}

FORMULAS = {
    "uniform": "std * sqrt(12) * (U(0,1) - 1/2)",
    "bernoulli": "std * (Bernoulli(p) - p) / sqrt(p*(1-p))",
    "beta": "std * (Beta(a,b) - a/(a+b)) / sqrt(a*b/((a+b)^2*(a+b+1)))",
    "binomial": "std * (Binomial(n,p) - n*p) / sqrt(n*p*(1-p))",
    "exponential": "std * (Exponential(scale=1) - 1)",
    "cauchy": "scale * Cauchy(location=0,scale=1); median=0, mean/variance undefined",
    "student_t": "std * StudentT(df) / sqrt(df/(df-2)); df > 2 for finite variance",
    "chi_square": "std * (ChiSquare(df) - df) / sqrt(2*df)",
    "f_distribution": "std * (F(d1,d2) - d2/(d2-2)) / sqrt(2*d2^2*(d1+d2-2)/(d1*(d2-2)^2*(d2-4))); d2 > 4",
    "gamma": "std * (Gamma(shape=k,scale=1) - k) / sqrt(k)",
    "weibull": "std * (Weibull(shape=k,scale=1) - Gamma(1+1/k)) / sqrt(Gamma(1+2/k)-Gamma(1+1/k)^2)",
    "laplace": "Laplace(location=0,scale=std/sqrt(2))",
    "logistic": "Logistic(location=0,scale=std*sqrt(3)/pi)",
}


def sample_distribution(name, p, rng, shape):
    """Return the unmasked field; one independent draw per pixel/channel.

Validated parameters come from noise.parameters(). std is a population field
standard deviation BEFORE gains/masking/shared cap, not achieved pixel RMS.
Heavy-tail samples are not truncated, resampled, blurred, or divided by their
observed maximum. The existing renderer alone clips the combined residual.
"""
    if name == "uniform":
        field = math.sqrt(12) * (rng.uniform(0, 1, shape) - 0.5)
    elif name == "bernoulli":
        prob = p["p"]
        field = (rng.binomial(1, prob, shape) - prob) / math.sqrt(prob*(1-prob))
    elif name == "beta":
        a, b = p["alpha"], p["beta"]
        field = (rng.beta(a, b, shape) - a/(a+b)) / math.sqrt(a*b/((a+b)**2*(a+b+1)))
    elif name == "binomial":
        n, prob = p["n"], p["p"]
        field = (rng.binomial(n, prob, shape) - n*prob) / math.sqrt(n*prob*(1-prob))
    elif name == "exponential":
        field = rng.exponential(1, shape) - 1
    elif name == "cauchy":
        return p["scale"] * rng.standard_cauchy(shape)
    elif name == "student_t":
        df = p["df"]
        field = rng.standard_t(df, shape) / math.sqrt(df/(df-2))
    elif name == "chi_square":
        df = p["df"]
        field = (rng.chisquare(df, shape) - df) / math.sqrt(2*df)
    elif name == "f_distribution":
        a, b = p["dfnum"], p["dfden"]
        var = 2*b*b*(a+b-2)/(a*(b-2)**2*(b-4))
        field = (rng.f(a, b, shape) - b/(b-2)) / math.sqrt(var)
    elif name == "gamma":
        k = p["shape"]
        field = (rng.gamma(k, 1, shape) - k) / math.sqrt(k)
    elif name == "weibull":
        k = p["shape"]
        mean = math.gamma(1+1/k)
        sd = math.sqrt(math.gamma(1+2/k)-mean*mean)
        field = (rng.weibull(k, shape) - mean) / sd
    elif name == "laplace":
        field = rng.laplace(0, 1/math.sqrt(2), shape)
    elif name == "logistic":
        field = rng.logistic(0, math.sqrt(3)/math.pi, shape)
    else:
        raise ValueError(f"Unknown distribution: {name}")
    return p["std"] * field


def distribution_catalog(name):
    return {
        "units": "shape/probability parameters are dimensionless; std/scale are field units before epsilon, gains and mask",
        "field_formula": FORMULAS[name],
        "sampling": "seeded NumPy RNG; independent additive samples per native pixel and RGB channel; image-independent",
        "normalization": "analytical population moments only (Cauchy: zero location); no sample-wise normalization or tail truncation",
        "projection": "sum all masked components then apply one shared pointwise L_inf cap; clipped/quantized residuals no longer follow the raw family exactly",
    }
