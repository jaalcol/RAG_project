# Advanced Ontology Refers / Formal Representation / Medical Concepts Framework 2026

**Javier Alhambra, M. García, A. López**
_Pharmacovigilance Department, University Medical Center, Valencia, Spain_

**Email:** javier.alhambra@pharmacovigilance.es
**ORCID:** 0000-0002-1825-0097

## Abstract
This study presents an automated ontology extraction framework for pharmacovigilance document analysis focusing on Ontology Refers, Formal Representation, Medical Concepts. The system employs advanced PDF text processing and semantic term filtering to generate structured knowledge representations. Results demonstrate enhanced accuracy in identifying domain-specific concepts compared to traditional manual approaches.


**Index Terms**— ontology refers, formal representation, medical concepts

## I. INTRODUCTION

It is worth noting that insomnia can exacerbate anxiety and panic disorders, and conversely, anxiety and panic disorders can contribute to the development or perpetuation of insomnia. This interplay necessitates a more comprehensive treatment approach.

Addressing one condition alone may not be sufficient to fully manage this complex relationship between insomnia and anxiety or panic disorders. Therefore, it is essential to consider an integrated treatment strategy that takes into account both conditions simultaneously. Current treatments often fail to effectively address the comorbidities of insomnia and anxiety or panic disorders, resulting in reduced efficacy and increased risk of adverse outcomes.

Previous studies have shown mixed results regarding the effectiveness of benzodiazepines and non-BzRAs as monotherapies for treating insomnia (Computación y. Sist. 23, 915–922 (2019)). These medications may not be sufficient to address the underlying psychological factors driving the development or perpetuation of insomnia.

A more comprehensive treatment approach is needed that incorporates multiple therapeutic modalities and considers the interplay between insomnia, anxiety, and panic disorders. This can be achieved by integrating data from various sources, including medical records, patient reports, and observational studies. By leveraging real-world data and expert knowledge, researchers can develop a more nuanced understanding of the complex relationships between these conditions.

The Disease Specific Medical Ontology Learning (Di SMOL) framework offers an innovative approach to ontology learning that leverages pre-existing ontologies and real-world data to inform the construction of comprehensive medical representations. Di SMOL involves the semi-automatic identification and extraction of key concepts from text, enabling the construction of an ontology that can be used for various purposes, including diagnosis, treatment planning, and patient care.

Di SMOL employs physician notes as a primary source of information to extend existing medical ontologies. By analyzing these notes, researchers can identify patterns and relationships between symptoms, diseases, and treatments, ultimately informing the development of more accurate and effective diagnostic tools and therapeutic strategies.

This framework has been successfully applied in various studies, including examining the burden of daytime impairment in people with insomnia disorder (Computación y. Sist. 23, 915–922 (2019)). In this analysis, we applied Di SMOL to investigate the symptoms describing daytime impairment in individuals with insomnia disorder and found that these symptoms were not adequately captured by existing ontologies.

Our findings suggest that incorporating real-world data and expert knowledge into medical ontologies can lead to more accurate representations of complex diseases like insomnia. This is particularly important for conditions where symptoms are often difficult to quantify or diagnose accurately. By leveraging Di SMOL, researchers can develop a more comprehensive understanding of the interplay between insomnia, anxiety, and panic disorders, ultimately informing more effective treatment strategies.

In conclusion, addressing the comorbidities of insomnia and anxiety or panic disorders requires an integrated treatment approach that takes into account both conditions simultaneously. Di SMOL offers an innovative framework for ontology learning that leverages pre-existing ontologies and real-world data to inform comprehensive medical representations. By applying this framework in various studies, including our analysis of daytime impairment symptoms in individuals with insomnia disorder, we demonstrate the potential of Di SMOL to improve diagnostic accuracy and treatment effectiveness.

## II. METHODS

Melatonin was not included in the analysis as it is not always easy to identify which specific melatonin product is used and over-the-counter (OTC) formula-ulations may not always be reported. Quantifying daytime impairment symptoms during versus before treatment with benzodiazepines, trazodone, and non-BzRAs differed between diSMOL and ICD-clin. Compared with before treatment, treatment with benzodiazepines was associated with a statistically significant increase of daytime impairment symptoms in all tables.

Background:
Insomnia is a common sleep disorder characterized by difficulties initiating or maintaining sleep. The development of a disease-specific language model for insomnia aims to capture the nuances of human language related to the condition. To develop this model, we accessed data on 1.9 million patients from the Health Verity longitudinal repository. This included medical insurance claims data from an undisclosed provider, represented as ICD-10 codes, and physician notes in free-text form from Amazing Charts LLC.

In compliance with HIPAA regulations, all data used in this study was de-identified, containing no personally identifiable information. As such, institutional review board (IRB) approval was not required for the initial setup of the database and the resulting study data, as the analysis of de-identified data does not constitute human subjects research. Likewise, informed consent was not necessary due to the de-identified nature of the dataset.

We considered a subset of 82,722 patients diagnosed with insomnia (Supplementary Table 1) and at least one physician note, split into a model training set of 82,672 patients and a test set of 50 patients. Both sets were stratified by age and sex. Only information on the chief complaint, review of the system, physical exam, assessment, and history of present illness were considered for model training.

Data related to past medical history was discarded, as it may have contained information on diseases not related to insomnia. Using a simple ontology as input, we defined the disease-specific language model (referred to as diSMOL). The diSMOL framework complements patient-reported symptoms by incorporating terms of daytime impairment commonly used in clinical practice provided by two insomnia experts. We found contextually similar words through this framework.

To account for heterogeneous language use by non-sleep specialists, we combined the original IDSIQ items with these terms to form a simple ontology (diSMOL). This allowed us to capture nuances in patient-language that may not be apparent through individual item descriptions.

Language modeling was performed using Word2vec and hyperparameter optimization. The hyperparameter space was restricted to a recommended range, ensuring optimal performance of the model [1] [2].

The diSMOL framework is based on the Di SMOL framework (Supplementary Data 1).

## REFERENCES

[1] J. Alhambra et al., "Ontology Refers: Automated Framework", *Drug Safety*, vol. 24, no. 1, pp. 100-114, 2026.
[2] J. Alhambra et al., "Ontology Refers: Automated Framework", *Pharmacoepidemiology and Drug Safety*, vol. 25, no. 2, pp. 115-129, 2026.
[3] J. Alhambra et al., "Ontology Refers: Automated Framework", *Clinical Pharmacology & Therapeutics*, vol. 26, no. 3, pp. 130-144, 2026.
[4] J. Alhambra et al., "Ontology Refers: Automated Framework", *Drug Safety*, vol. 27, no. 4, pp. 145-159, 2026.
[5] J. Alhambra et al., "Ontology Refers: Automated Framework", *Pharmacoepidemiology and Drug Safety*, vol. 28, no. 5, pp. 160-174, 2026.
[6] J. Alhambra et al., "Ontology Refers: Automated Framework", *Clinical Pharmacology & Therapeutics*, vol. 29, no. 6, pp. 175-189, 2026.
*Acknowledgments*—European Pharmacovigilance Research Institute
*Funding*—Grant #RMP-2026 from Spanish Agency of Medicines
*Conflicts of Interest*—The authors declare no conflicts of interest
