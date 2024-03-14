[Text generation strategies (huggingface.co)](https://huggingface.co/docs/transformers/generation_strategies#speculative-decoding)

docker run --ipc=host --network=host --privileged -itd --name xx-dis --gpus device=all -v /data/xx/dis:/dis nvcr.io/nvidia/pytorch:23.09-py3

docker exec --privileged -it xx-dis /bin/bash 

```
/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py(1217)generate()  
:1504
result = self.assisted_decoding(
model_kwargs
{'attention_mask': tensor([[1, 1, 1, 1]]), 'output_attentions': False, 'output_hidden_states': False, 'use_cache': True}
candidate_generator = AssistedCandidateGenerator(

/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py(4300)assisted_decoding()
unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
while True:
    candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids)#[1,prompt+new],[1,new,50304]#有kv cache
    model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
    outputs = self(
                **model_inputs,#input_ids[1,prompt+new]
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )#有kv cache
    n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
    candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)#增减num_assistant_tokens
```

one seq one prefill not batch

batch seq?

