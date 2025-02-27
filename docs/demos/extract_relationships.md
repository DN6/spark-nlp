---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /extract_relationships
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Healthcare 
      excerpt: Extract Relationships 
      secheader: yes
      secheader:
        - title: Spark NLP for Healthcare
          subtitle: Extract Relationships 
          activemenu: extract_relationships
      source: yes
      source: 
        - title: Detect posology relations
          id: detect_posology_relations
          image: 
              src: /assets/images/Grammar_Analysis.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: Automatically identify relations between drugs, dosage, duration, frequency and strength using our pretrained clinical Relation Extraction (RE) model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_POSOLOGY/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_POSOLOGY.ipynb
        - title: Detect temporal relations for clinical events
          id: detect_temporal_relations_for_clinical_events
          image: 
              src: /assets/images/Grammar_Analysis.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: 'Automatically identify three types of relations between clinical events: After, Before and Overlap using our pretrained clinical Relation Extraction (RE) model.'
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL_EVENTS/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_CLINICAL_EVENTS.ipynb
        - title: Detect causality between symptoms and treatment
          id: detect_causality_between_symptoms_and_treatment
          image: 
              src: /assets/images/Grammar_Analysis.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: Automatically identify relations between symptoms and treatment using our pretrained clinical Relation Extraction (RE) model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_CLINICAL.ipynb
        - title: Detect relations between body parts and clinical entities
          id: detect_relations_between_body_parts_and_clinical_entities
          image: 
              src: /assets/images/Detect_relations.svg
          image2: 
              src: /assets/images/Detect_relations_f.svg
          excerpt: Use pre-trained relation extraction models to extract relations between body parts and clinical entities.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_BODYPART_ENT/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.1.Clinical_Relation_Extraction_BodyParts_Models.ipynb
        - title: Detect how dates relate to clinical entities
          id: detect_how_dates_relate_to_clinical_entities
          image: 
              src: /assets/images/ExtractRelationships_2.svg
          image2: 
              src: /assets/images/ExtractRelationships_2_f.svg
          excerpt: Detect clinical entities such as problems, tests and treatments, and how they relate to specific dates.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL_DATE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.1.Clinical_Relation_Extraction_BodyParts_Models.ipynb
        - title: Detect drugs interactions
          id: detect_drugs_interactions
          image: 
              src: /assets/images/Detect_drugs_interactions.svg
          image2: 
              src: /assets/images/Detect_drugs_interactions_c.svg
          excerpt: Detect possible interactions between drugs using out-of-the-box Relation Extraction Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_DRUG_DRUG_INT/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb
        - title: Detect relations between chemicals and proteins
          hide: yes
          id: detect_relations_between_chemicals_and_proteins
          image: 
              src: /assets/images/Detect_relations_between_chemicals_and_proteins.svg
          image2: 
              src: /assets/images/Detect_relations_between_chemicals_and_proteins_c.svg
          excerpt: Automatically detect possible relationships between chemicals and proteins using a predefined Relation Extraction model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_CHEM_PROT/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb
        - title: Identify relations between scale items and measurements according to NIHSS
          id: identify_relations_between_scale_items_their_measurements_according
          image: 
              src: /assets/images/Identify_relations_between_scale_items_measurements_according.svg
          image2: 
              src: /assets/images/Identify_relations_between_scale_items_measurements_according_f.svg
          excerpt: This demo shows how relations between scale items and their measurements can be identified according to NIHSS guidelines using a Spark NLP Healthcare RE model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_NIHSS/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb
        - title: Identify Relations Between Drugs and ADE
          id: identify_relations_between_drugs_and_ade
          image: 
              src: /assets/images/Identify_relations_between_drugs_and_ade.svg
          image2: 
              src: /assets/images/Identify_relations_between_drugs_and_ade_f.svg
          excerpt: This demo shows how to detect relations between drugs and adverse reactions caused by them.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_ADE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb
---
