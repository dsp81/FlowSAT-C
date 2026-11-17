from metacf.tier1.t1_recovery import T1Recovery
from metacf.tier1.t2_sensitivity import T2Sensitivity
from metacf.tier1.t3_monotonic import T3Monotonic
from metacf.tier1.t4_calibration import T4Calibration
from metacf.tier1.t5_resolution import T5Resolution
from metacf.tier1.t6_specificity import T6Specificity
from metacf.tier1.t7_memorization import T7Memorization
from metacf.tier1.t8_diversity import T8Diversity
from metacf.tier1.t9_pathways import T9Pathways
from metacf.tier1.t10_prompt import T10PromptCompetition

ALL = [T1Recovery, T2Sensitivity, T3Monotonic, T4Calibration, T5Resolution, T6Specificity,
       T7Memorization, T8Diversity, T9Pathways, T10PromptCompetition]
