import base64
import json
import struct
import os

from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment
from similarity_tools.helpers.constants import DST_DATA_DIR, PIPELINE_SUBDIRECTORY

es_view_id = "https://staging.nise.bbp.epfl.ch/views/SarahTest/PublicThalamusTest2/7e09b95c-e2c7-40d4-aedd-f0358740052e"

forge = NexusBucketConfiguration(
    organisation="SarahTest", project="PublicThalamusTest2",
    deployment=Deployment.STAGING, elastic_search_view=es_view_id
).allocate_forge_session()

embedding_resources = forge.search({"type": "Embedding"}, limit=200, search_endpoint="elastic")

org, project = "public", "thalamus"

model_path = f"morph_TMD_image_data_100_{org}_{project}_production.json"


# Get vector from initial model dict
with open(model_path, "r") as f:
    model_content = json.load(f)

model_path2 = f"morph_TMD_image_data_base64_100_{org}_{project}_production.json"

# Get vector from encoded initial model dict
with open(model_path2, "r") as f:
    model_content_encoded = json.load(f)


# https://stackoverflow.com/questions/57211362/how-to-convert-binary-data-back-to-a-float-array-in-elasticsearch-painless

script_source = """
    
    byte[] toBytes(String x) { 
    
        int length = x.length();
        
        char[] buffer = new char[length];
    
        x.getChars(0, length, buffer, 0);
    
        byte[] b = new byte[length];
        
        for (int j = 0; j < length; j++){
            b[j] = (byte) buffer[j];
        } 
        return b;
    }
    
    float[] toFloat(byte[] arr) {
    
        int length = arr.length/4;
    
        float[] vector = new float[length];
    
        for (int i = 0; i < length; ++i) {
            def n = i*4;
            vector[i] = Float.intBitsToFloat( (arr[n+3] << 24) | ((arr[n+2] & 255) << 16) |  ((arr[n+1] & 255) << 8) |  (arr[n] & 255) );   
        }
        
        return vector;
    }

    float[] vector = toFloat(doc["embedding"].value.bytes);
    
    float[] q_vector = toFloat(Base64.getDecoder().decode(params.query_vector));

    float distance = 0;
    
    for (int i = 0; i < q_vector.length; ++i) {
        distance += Math.abs(vector[i] - q_vector[i]);
    }

    return 1/(1+distance);
"""

vector_parsed_script = """
    def vector_bytes = doc["embedding"].value.bytes;
    def vector = new float[vector_bytes.length/4];
    for (int i = 0; i < vector.length; ++i) {
      def n = i*4;
      vector[i] = Float.intBitsToFloat( (vector_bytes[n+3] << 24) | ((vector_bytes[n+2] & 255) << 16) |  ((vector_bytes[n+1] & 255) << 8) |  (vector_bytes[n] & 255) );
    }
    return vector;
"""

def query_check(embedding_resource):
    embedding = embedding_resource.embedding if not isinstance(embedding_resource.embedding, list) else embedding_resource.embedding[0]
    embedding_derivation = next(i for i in embedding_resource.derivation if "NeuronMorphology" in i.entity.type).entity.id


    query = {
        "from": 0,
        "size": 10,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        # "must_not": {
                        #     "term": {"@id": embedding_resource.id}
                        # },
                        "must": [{
                            "exists": {"field": "embedding"}
                        }]
                    }
                },
                "script": {
                    "source": script_source,
                    "params": {
                        "query_vector": embedding
                    }
                }
            },
        },
        "script_fields": {
            "vector_parsed": {
                "script": {
                    "source": vector_parsed_script
                }
            }
        },
        "_source": ["derivation", "embedding"]
    }

    res = forge.elastic(json.dumps(query), as_resource=False)

    res_i = res[0]
    derivation = next(i for i in res_i["_source"]["derivation"] if "NeuronMorphology" in i["entity"]["@type"])["entity"]["@id"]

    # print("Initial embedding derivation", embedding_derivation)
    # print("First result derivation", derivation)
    # print([re["_score"] for re in res])

    encoded_local = model_content_encoded[next(i for i in model_content_encoded.keys() if i.startswith(derivation))]

    assert encoded_local == res_i["_source"]["embedding"]

    # initial_vector = model_content[next(i for i in model_content.keys() if i.startswith(derivation))]
    #
    # # Get vector parsed from ES
    # parsed = res_i['fields']['vector_parsed'][0]
    #
    # # Get vector from ES and parse it with python
    # decoded = base64.b64decode(res_i["_source"]["embedding"])
    # parsed_python = struct.unpack(f'{len(initial_vector)}f', decoded)
    #
    # try:
    #     for i in range(10000):
    #         assert round(parsed[i], 5) == round(parsed_python[i], 5)
    #         assert round(parsed_python[i], 5) == round(initial_vector[i], 5)
    # except AssertionError:
    #     print(f"Failed assertion for {derivation}")

    return derivation == embedding_derivation


check_over_all = [query_check(embedding_resource) for embedding_resource in embedding_resources]
print(len([i for i in check_over_all if i]))
