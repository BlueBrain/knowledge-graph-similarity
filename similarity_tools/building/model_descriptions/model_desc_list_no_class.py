from similarity_tools.data_classes.model_description import ModelDescription

axon_model_description = ModelDescription({
    "name": "NeuronMorphology Axon Co-Projection Embedding",
    "description": "Node embedding model (node2vec) built on an axon co-projection graph "
                   "extracted from a neuron morphology dataset",
    "filename": "morph_axon_coproj_node2vec_cosine",
    "label": "Axon projection",
    "distance": "cosine",
    "model": None,
    "rev": None
})


bbp_brain_region_model_description = ModelDescription({
    "name": "NeuronMorphology Brain Region Embedding - BBP Mouse Brain region ontology",
    "description": "Poincare node embedding of brain regions in BBP Mouse Brain region ontology",
    "filename": "morph_brain_region_poincare_bbp",
    "label": "Brain regions BBP Mouse Brain region ontology",
    "distance": "poincare",
    "model": None,
    "rev": None
})

bbp_brain_region_bmo_model_description = ModelDescription({
    "name": "NeuronMorphology Brain Region Embedding - BBP Mouse Brain region ontology - BMO File",
    "description":
        "Poincare node embedding of brain regions in BBP Mouse Brain region ontology - BMO File",
    "filename": "morph_brain_region_bmo_poincare_bbp",
    "label": "Brain regions BBP Mouse Brain region ontology - BMO File",
    "distance": "poincare",
    "model": None,
    "rev": None
})

allen_brain_region_model_description = ModelDescription({
    "name": "NeuronMorphology Brain Region Embedding",
    "description": "Poincare node embedding of brain regions in Allen CCFv3 of a neuron"
                   " morphology dataset Brain regions (CCfv3)",
    "filename": "morph_brain_region_poincare_allen",
    "label": "Brain regions (CCfv3)",
    "distance": "poincare",
    "model": None,
    "rev": None
})


bbp_brain_region_alone_model_description = ModelDescription({
    "name": "BBP Brain Region Ontology Embedding",
    "description": "Poincare node embedding of brain regions in BBP Mouse Brain region ontology",
    "filename": "brain_region_poincare_bbp",
    "label": "BBP Brain Region Ontology Embedding",
    "distance": "poincare",
    "model": None,
    "rev": None
})

coordinate_model_description = ModelDescription({
    "name": "NeuronMorphology Coordinates",
    "description": "Coordinate similarity of a neuron morphology dataset",
    "filename": "morph_coordinates_euclidean",
    "label": "Coordinates",
    "distance": "euclidean",
    "model": None,
    "rev": None
})

dendrite_model_description = ModelDescription({
    "name": "NeuronMorphology Dendrite Co-Projection Embedding",
    "description": "Node embedding model (node2vec) built on a dendrite co-projection graph "
                   "extracted from a neuron morphology dataset",
    "filename": "morph_dendrite_coproj_node2vec_cosine",
    "label": "Dendrite projection",
    "distance": "cosine",
    "model": None,
    "rev": None
})

unscaled_model_description = ModelDescription({
    "name": "NeuronMorphology TMD-based Embedding",
    "description": "Vectorization of unscaled persistence diagrams of neuron morphologies",
    "filename": "morph_TMD_euclidean",
    "label": "TMD",
    "distance": "euclidean",
    "model": None,
    "rev": None
})

scaled_model_description = ModelDescription({
    "name": "NeuronMorphology scaled TMD-based Embedding",
    "description": "Vectorization of scaled persistence diagrams of neuron morphologies",
    "filename": "morph_scaled_TMD_euclidean",
    "label": "TMD (scaled)",
    "distance": "euclidean",
    "model": None,
    "rev": None
})

neurite_model_description = ModelDescription({
    "name": "NeuronMorphology Neurite Features",
    "description": "Feature encoding model built for neurite features from neuron "
                   "morphology dataset resources Neurite features (including apical dendrite)",
    "filename": "morph_neurite_features_euclidean_apical",
    "label": "Neurite features",
    "distance": "euclidean",
    "model": None,
    "rev": None
})

new_tmd_model_description = ModelDescription({
    "name": "NeuronMorphology TMD-based Embedding of basal dendrite tree",
    "description": "Vectorization of persistence diagrams of neuron morphologies' basal dendrite",
    "filename": "morph_TMD_image_data",
    "label": "TMD",
    "distance": "custom_tmd",
    "model": None,
    "rev": None
})
