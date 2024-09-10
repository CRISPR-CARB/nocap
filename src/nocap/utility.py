import networkx as nx
import tellurium as te
import libsbml

def create_sbml_model_from_nx(dag, output_file='model.xml', hill_params=None):
    def dag_to_antimony(dag, hill_params):
        # Default Hill parameters
        default_hill_params = {
            'beta': 1.0,
            'K': 1.0,
            'n': 1.0
        }
        
        if hill_params is None:
            hill_params = {}

        # Check if the input graph is a directed acyclic graph (DAG)
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Input graph must be a directed acyclic graph (DAG).")

        model_str = ""
        model_str += "model grn_model()\n"
        # Global compartment declaration
        model_str += "  compartment default_compartment = 1;\n"

        # Check and retrieve initial values for nodes

        node_names = [node[0] for node in dag.nodes(data=True)]
        node_initial_values = [node[1]['value'] for node in dag.nodes(data=True)]
        last_node_name = node_names[-1]

        # total_initial_values = sum(node_initial_values)
        # n_1_names_str = " + ".join(node_names[:-1])


        for node in dag.nodes(data=True):
            #print(node)
            value = node[1].get('value', None)
            if value is None:
                raise ValueError(f"Node {node[0]} is missing the 'value' attribute.")
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Node {node[0]} has an invalid 'value' attribute: {value}. Must be a non-negative number.")

            model_str += f"  species {node[0]} in default_compartment;\n"

            if node[0] == last_node_name:  # constraint
                pass
                #model_str += f"  {node[0]} := {total_initial_values} - ({n_1_names_str});\n"
                #model_str += f"  const {node[0]};\n"
                model_str += f"  {node[0]} = {value};\n"
            else: 
                model_str += f"  {node[0]} = {value};\n"

            # # Update the nodes to mark source nodes as constant
            # source_nodes = [node for node in dag.nodes() if dag.in_degree(node) == 0]
            # for source_node in source_nodes:
            #     model_str += f"  const {source_node};\n"

        for source, target, data in dag.edges(data=True):
            if 'interaction_type' not in data:
                raise ValueError(f"No 'interaction_type' attribute for edge {source} -> {target}")

            interaction_type = data['interaction_type']
            beta = hill_params.get(f'beta_{source}_to_{target}', default_hill_params['beta'])
            K = hill_params.get(f'K_{source}_to_{target}', default_hill_params['K'])
            n = hill_params.get(f'n_{source}_to_{target}', default_hill_params['n'])
            
            if interaction_type == 'activation':
                rate_law = f"{beta} * {source}^n_{source}_to_{target} / (K_{source}_to_{target}^n_{source}_to_{target} + {source}^n_{source}_to_{target})"
                model_str += f"  J_{source}_to_{target}: {source} => {target}; {rate_law};\n"
            elif interaction_type == 'repression':
                rate_law = f"{beta} / (1 + ({source}^n_{source}_to_{target} / {K}))"
                model_str += f"  J_{source}_to_{target}: {source} => {target}; {rate_law};\n"
            else:
                raise ValueError(f"Unknown interaction type: {interaction_type}")

            # Add parameters
            model_str += f"  beta_{source}_to_{target} = {beta}; // units: mole_per_second\n"
            model_str += f"  K_{source}_to_{target} = {K}; // units: mole\n"
            model_str += f"  n_{source}_to_{target} = {n}; // units: dimensionless\n"

        model_str += "end"

        return model_str

    antimony_str = dag_to_antimony(dag, hill_params)
    #print("Generated Antimony Model:")
    #print(antimony_str)

    # Load the model into Tellurium and convert to SBML
    try:
        r = te.loada(antimony_str)
    except Exception as e:
        print(f"Error in Antimony model: {e}")
        raise
    sbml_str = r.getSBML()

    # Post-process the SBML with libSBML to ensure correct configurations
    document = libsbml.readSBMLFromString(sbml_str)
    model = document.getModel()

    if model is not None:
        for species in model.getListOfSpecies():
            initial_value = species.getInitialConcentration()  # Get the initial concentration from Antimony
            species.setInitialAmount(0.0)  # Set initial amount to 0.0
            species.setInitialConcentration(initial_value)  # Set initial concentration
            species.setHasOnlySubstanceUnits(False)
            species.setName(species.getId())

    
        # Write updated SBML back to string
        sbml_str = libsbml.writeSBMLToString(document)



    # Save the updated SBML to a file
    with open(output_file, "w") as f:
        f.write(sbml_str)

    print(f"SBML model written to {output_file}")