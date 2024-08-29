import json


class CustomSequentialChain:
    def __init__(self):
        self.chains = []
        self.chain_keys = []
        self.chain_inputs, self.chain_outputs = [], []
        self.run_flag = False

    def add_chain(self, llm, prompt, chain_key, response_schema):
        # Create a chain (and append it)
        chain_new = prompt | llm
        self.chains.append(chain_new)

        # Set inputs/outputs
        self.chain_keys.append(chain_key)
        self.chain_outputs.append({var.name: None for var in response_schema})
        self.chain_inputs.append({var: None for var in prompt.input_variables})

        # Re-instantiate the flag
        self.run_flag = False

    def run(self, verbose: bool = False, include_all_outputs: bool = False, **all_inputs):
        # Fill in the input values from the given input values
        for chain_input in self.chain_inputs:  # Go through all input dicts
            for input_variable_name in chain_input:  # Go through all inputs per dict
                if input_variable_name in all_inputs:  # Only add the ones that exist in the current input
                    chain_input[input_variable_name] = all_inputs[input_variable_name]

        # Run the chain
        for i, chain in enumerate(self.chains):
            # Logging
            if verbose:
                print(f"Running chain:{self.chain_keys[i]} with inputs: {self.chain_inputs[i]}.")

            # Run chain
            output_str = chain.invoke(input=self.chain_inputs[i]).content.strip("```json").strip("```").strip()
            outputs = json.loads(output_str)

            # Save outputs
            for output_var_name in self.chain_outputs[i]:
                self.chain_outputs[i][output_var_name] = outputs[output_var_name]

            # Fill in future inputs
            for future_input in self.chain_inputs[(i+1):]:
                for output_var_name in outputs:
                    if output_var_name in future_input:
                        future_input[output_var_name] = outputs[output_var_name]
            if verbose:
                print(f"Output of chain:{self.chain_keys[i]} is: {outputs}.\n")

        # Note that the chain was run
        self.run_flag = True

        # Return outputs
        if include_all_outputs:
            all_outputs = {}
            for outputs in self.chain_outputs:
                all_outputs.update(outputs)
            return all_outputs
        return self.chain_outputs[-1]

    def get_chain_inputs(self):
        if not self.run_flag:
            raise LookupError("Chain first needs to be run before getting the inputs!")
        return self.chain_inputs

    def get_chain_outputs(self):
        if not self.run_flag:
            raise LookupError("Chain second needs to be run before getting the outputs!")
        return self.chain_outputs

    def get_chain_input_by_index(self, index: int):
        if index not in self.chain_inputs:
            raise ValueError(f"Chain index:{index} is not found!")
        return self.chain_inputs[index]

    def get_chain_input_by_key(self, key: str):
        if key not in self.chain_keys:
            raise ValueError(f"Chain key:{key} is not found!")
        return self.chain_inputs[self.chain_keys.index(key)]

    def get_chain_output_by_index(self, index: int):
        if index not in self.chain_outputs:
            raise ValueError(f"Chain index:{index} is not found!")
        return self.chain_outputs[index]

    def get_chain_output_by_key(self, key: str):
        if key not in self.chain_keys:
            raise ValueError(f"Chain key:{key} is not found!")
        return self.chain_outputs[self.chain_keys.index(key)]
