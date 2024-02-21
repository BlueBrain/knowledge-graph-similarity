from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration


new_tag = "v4"
test = NexusBucketConfiguration("bbp", "inference-rules", True).allocate_forge_session()

a = "https://bbp.epfl.ch/neurosciencegraph/data/5d04995a-6220-4e82-b847-8c3a87030e0b"  # hierarchy
b = "https://bbp.epfl.ch/neurosciencegraph/data/abb1949e-dc16-4719-b43b-ff88dabc4cb8"  # neuron m
c = "https://bbp.epfl.ch/neurosciencegraph/data/9d64dc0d-07d1-4624-b409-cdc47ccda212" # gen sim br
to_tag = [test.retrieve(el) for el in [a, b, c]]
test.tag(to_tag, new_tag)
