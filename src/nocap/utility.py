import networkx as nx
import tellurium as te
import libsbml

def create_sbml_model_from_nx(dag, output_file='model.xml'):
    def dag_to_antimony(dag):

        # Check if the input graph is a directed acyclic graph (DAG)
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Input graph must be a directed acyclic graph (DAG).")

        model_str = ""

        model_str += "model grn_model()\n"
        # Global compartment declaration
        model_str += "  compartment default_compartment = 1;\n"

        # Check and retrieve initial values for nodes
        for node in dag.nodes(data=True):
            value = node[1].get('value', None)
            if value is None:
                raise ValueError(f"Node {node[0]} is missing the 'value' attribute.")
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Node {node[0]} has an invalid 'value' attribute: {value}. Must be a non-negative number.")

            model_str += f"  species {node[0]} in default_compartment;\n"
            model_str += f"  {node[0]} = {value};\n"

        for source, target, data in dag.edges(data=True):
            if 'interaction_type' not in data:
                raise ValueError(f"No 'interaction_type' attribute for edge {source} -> {target}")

            interaction_type = data['interaction_type']
            if interaction_type == 'activation':
                rate_law = f"beta_{source}_to_{target} * {source}^n_{source}_to_{target} / (K_{source}_to_{target}^n_{source}_to_{target} + {source}^n_{source}_to_{target})"
                model_str += f"  J_{source}_to_{target}: {source} => {target}; {rate_law};\n"
            elif interaction_type == 'repression':
                rate_law = f"beta_{source}_to_{target} / (1 + ({source}^n_{source}_to_{target} / K_{source}_to_{target}^n_{source}_to_{target}))"
                model_str += f"  J_{source}_to_{target}: {source} => {target}; {rate_law};\n"
            else:
                raise ValueError(f"Unknown interaction type: {interaction_type}")

            # Add parameters and units
            model_str += f"  beta_{source}_to_{target} = 1.0; // units: mole_per_second\n"
            model_str += f"  K_{source}_to_{target} = 1.0; // units: mole\n"
            model_str += f"  n_{source}_to_{target} = 1.0; // units: dimensionless\n"

        model_str += "end"

        return model_str

    antimony_str = dag_to_antimony(dag)
    print("Generated Antimony Model:")
    print(antimony_str)

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
