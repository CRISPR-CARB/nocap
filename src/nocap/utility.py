import networkx as nx
import tellurium as te
import libsbml

def create_sbml_model_from_nx(dag, output_file='model.xml'):
    def dag_to_antimony(dag):
        model_str = ""

        model_str += "model grn_model()\n"
        # Global compartment declaration
        model_str += "  compartment default_compartment = 1;\n"

        for node in dag.nodes():
            model_str += f"  species {node} in default_compartment;\n"
            model_str += f"  {node} = 1;\n"

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
            species.setInitialAmount(0.0)  # Set initial amount to 0.0
            species.setHasOnlySubstanceUnits(False)
            species.setInitialConcentration(1.0)  # Ensure initial concentration is correct
            species.setName(species.getId())

        # for compartment in model.getListOfCompartments():
        #     compartment.setUnits("litre")  # Ensure the units are set to litre for compartments
        
        # Write updated SBML back to string
        sbml_str = libsbml.writeSBMLToString(document)

    # Save the updated SBML to a file
    with open(output_file, "w") as f:
        f.write(sbml_str)

    print(f"SBML model written to {output_file}")

