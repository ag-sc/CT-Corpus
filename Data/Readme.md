## Content:

**1. Corpus.zip**

- list_abstracts_glaucoma_corpus: list of the PubMed ids of the abstracts of clinical trials on glaucoma

- list_abstracts_dm2_corpus: list of the PubMed ids of the abstracts of clinical trials on type 2 diabetes mellitus

- glaucoma_corpus: corpus of glaucoma abstracts (original text, text without newlines and tokenized text)

- diabetes_corpus: corpus of glaucoma abstracts (original text, text without newlines and tokenized text)


**2. AnnotatedCorpus.zip**

- Annotated abstracts on glaucoma and diabetes:

- Slot-annotated_for_IAA: contains 20 schematically annotated abstracts (slot-filling templates) abstracts and their corresponding single-entity annotated files used for the inter-annotator agreement on complex entities. The 20 files were annotated by different annotators that the ones contained in the Slot-annotated subdirectory.

- Slot-annotated: contains the slot-filling template annotations in RDF n-triple format.

- Entity-annotated: contains the single-entity annotated abstracts in CONLL fashion format with the elements: *AnnotationID, ClassType (annotation category), DocCharOnset(incl), DocCharOffset(excl), Text (annotated text), Meta (N/A), Instances (RDF triple indicated when the annotation is a slot-filler).*

- Tokens: contains tokenized files.

"annotator[1-4]" corresponds to each annotator and "curated" to the curated files.


**3. AnnotationGuidelines.pdf**

Annotation guidelines for the entity and schematic annotation of clinical trial abstracts.
