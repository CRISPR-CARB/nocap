from nocap.utility import create_sbml_model_from_nx

import networkx as nx
import libsbml
import os

def test_create_sbml_model_from_nx():
    # Example DAG 1
    dag1 = nx.DiGraph()
    dag1.add_edge('GeneA', 'GeneB', interaction_type='activation')
    dag1.add_edge('GeneB', 'GeneC', interaction_type='repression')

    # Example DAG 2
    dag2 = nx.DiGraph()
    dag2.add_edge('Gene1', 'Gene2', interaction_type='activation')
    dag2.add_edge('Gene2', 'Gene3', interaction_type='activation')
    dag2.add_edge('Gene3', 'Gene4', interaction_type='repression')

    # Test case where a parent has multiple descendants
    dag3 = nx.DiGraph()
    dag3.add_edge('GeneA', 'GeneB', interaction_type='activation')
    dag3.add_edge('GeneC', 'GeneB', interaction_type='activation')
    dag3.add_edge('GeneB', 'GeneD', interaction_type='repression')
    
    test_cases = [
        (
            dag1,
            ['GeneA', 'GeneB', 'GeneC'],
            {
                'J_GeneA_to_GeneB': 'beta_GeneA_to_GeneB * GeneA^n_GeneA_to_GeneB / (K_GeneA_to_GeneB^n_GeneA_to_GeneB + GeneA^n_GeneA_to_GeneB)',
                'J_GeneB_to_GeneC': 'beta_GeneB_to_GeneC / (1 + GeneB^n_GeneB_to_GeneC / K_GeneB_to_GeneC^n_GeneB_to_GeneC)'
            },
            {
                'beta_GeneA_to_GeneB': 1.0,
                'K_GeneA_to_GeneB': 1.0,
                'n_GeneA_to_GeneB': 1.0,
                'beta_GeneB_to_GeneC': 1.0,
                'K_GeneB_to_GeneC': 1.0,
                'n_GeneB_to_GeneC': 1.0
            },
            {
                'GeneA': 1.0,
                'GeneB': 1.0,
                'GeneC': 1.0
            }
        ),
        (
            dag2,
            ['Gene1', 'Gene2', 'Gene3', 'Gene4'],
            {
                'J_Gene1_to_Gene2': 'beta_Gene1_to_Gene2 * Gene1^n_Gene1_to_Gene2 / (K_Gene1_to_Gene2^n_Gene1_to_Gene2 + Gene1^n_Gene1_to_Gene2)',
                'J_Gene2_to_Gene3': 'beta_Gene2_to_Gene3 * Gene2^n_Gene2_to_Gene3 / (K_Gene2_to_Gene3^n_Gene2_to_Gene3 + Gene2^n_Gene2_to_Gene3)',
                'J_Gene3_to_Gene4': 'beta_Gene3_to_Gene4 / (1 + Gene3^n_Gene3_to_Gene4 / K_Gene3_to_Gene4^n_Gene3_to_Gene4)'
            },
            {
                'beta_Gene1_to_Gene2': 1.0,
                'K_Gene1_to_Gene2': 1.0,
                'n_Gene1_to_Gene2': 1.0,
                'beta_Gene2_to_Gene3': 1.0,
                'K_Gene2_to_Gene3': 1.0,
                'n_Gene2_to_Gene3': 1.0,
                'beta_Gene3_to_Gene4': 1.0,
                'K_Gene3_to_Gene4': 1.0,
                'n_Gene3_to_Gene4': 1.0
            },
            {
                'Gene1': 1.0,
                'Gene2': 1.0,
                'Gene3': 1.0,
                'Gene4': 1.0
            }
        ),
        (
            dag3,
            ['GeneA', 'GeneB', 'GeneC', 'GeneD'],
            {
                'J_GeneA_to_GeneB': 'beta_GeneA_to_GeneB * GeneA^n_GeneA_to_GeneB / (K_GeneA_to_GeneB^n_GeneA_to_GeneB + GeneA^n_GeneA_to_GeneB)',
                'J_GeneC_to_GeneB': 'beta_GeneC_to_GeneB * GeneC^n_GeneC_to_GeneB / (K_GeneC_to_GeneB^n_GeneC_to_GeneB + GeneC^n_GeneC_to_GeneB)',
                'J_GeneB_to_GeneD': 'beta_GeneB_to_GeneD / (1 + GeneB^n_GeneB_to_GeneD / K_GeneB_to_GeneD^n_GeneB_to_GeneD)'
            },
            {
                'beta_GeneA_to_GeneB': 1.0,
                'K_GeneA_to_GeneB': 1.0,
                'n_GeneA_to_GeneB': 1.0,
                'beta_GeneC_to_GeneB': 1.0,
                'K_GeneC_to_GeneB': 1.0,
                'n_GeneC_to_GeneB': 1.0,
                'beta_GeneB_to_GeneD': 1.0,
                'K_GeneB_to_GeneD': 1.0,
                'n_GeneB_to_GeneD': 1.0
            },
            {
                'GeneA': 1.0,
                'GeneB': 1.0,
                'GeneC': 1.0,
                'GeneD': 1.0
            }
        )
    ]
    
    for dag, expected_species, expected_reactions, expected_parameters, expected_concentrations in test_cases:
        # Define the output file path
        output_file = 'test_model.xml'

        # Generate SBML model
        create_sbml_model_from_nx(dag, output_file)

        # Read SBML document from the file using libsbml
        document = libsbml.readSBML(output_file)

        # Test for valid SBML
        def test_valid_sbml(document):
            errors = document.checkConsistency()
            if errors > 0:
                print("---- SBML Errors ----")
                for i in range(errors):
                    print(document.getError(i).getMessage())
                print("---------------------")
            assert errors == 0, "SBML document is not valid"

        # Test for correct number of species and species IDs
        def test_species(document, expected_species):
            model = document.getModel()
            species_ids = [species.getId() for species in model.getListOfSpecies()]
            assert len(species_ids) == len(expected_species), "Number of species does not match"
            for species_id in expected_species:
                assert species_id in species_ids, f"Species {species_id} not found"

        # Test for correct number of reactions and reaction IDs
        def test_reactions(document, expected_reactions):
            model = document.getModel()
            reaction_ids = [reaction.getId() for reaction in model.getListOfReactions()]
            assert len(reaction_ids) == len(expected_reactions), "Number of reactions does not match"
            for reaction_id in expected_reactions:
                assert reaction_id in reaction_ids, f"Reaction {reaction_id} not found"

        # Test for correct compartment
        def test_compartment(document, expected_compartment_id):
            model = document.getModel()
            compartments = model.getListOfCompartments()
            assert len(compartments) == 1, "Number of compartments is not 1"
            assert compartments[0].getId() == expected_compartment_id, f"Compartment ID does not match"

        # Test for correct number of reactions and reaction equations
        def test_reaction_equations(document, expected_reactions):
            model = document.getModel()
            for reaction_id, expected_eq in expected_reactions.items():
                reaction = model.getReaction(reaction_id)
                assert reaction is not None, f"Reaction {reaction_id} not found"
                math_ast = reaction.getKineticLaw().getMath()
                formula = libsbml.formulaToL3String(math_ast)
                print(f"Equation for {reaction_id}: {formula}")
                assert formula == expected_eq, f"Equation for {reaction_id} does not match: {formula} != {expected_eq}"

        # Test for correct parameter values (rate constants)
        def test_parameters(document, expected_parameters):
            model = document.getModel()
            for reaction in model.getListOfReactions():
                kinetic_law = reaction.getKineticLaw()
                for param in kinetic_law.getListOfParameters():
                    param_id = param.getId()
                    value = param.getValue()
                    print(f"Parameter {param_id} value: {value}")
                    assert param_id in expected_parameters, f"Parameter {param_id} not expected"
                    assert value == expected_parameters[param_id], f"Value for {param_id} does not match: {value} != {expected_parameters[param_id]}"

        # Test for species concentrations
        def test_species_concentrations(document, expected_concentrations):
            model = document.getModel()
            for species_id, expected_concentration in expected_concentrations.items():
                species = model.getSpecies(species_id)
                assert species is not None, f"Species {species_id} not found"
                init_conc = species.getInitialConcentration()
                print(f"Initial concentration for {species_id}: {init_conc}")
                assert init_conc == expected_concentration, f"Concentration for {species_id} does not match: {init_conc} != {expected_concentration}"

        # Running the sub-tests
        #test_valid_sbml(document)
        test_species(document, expected_species)
        test_reactions(document, expected_reactions.keys())
        test_compartment(document, 'default_compartment')
        test_reaction_equations(document, expected_reactions)
        test_parameters(document, expected_parameters)
        test_species_concentrations(document, expected_concentrations)

        # Cleanup the generated file
        os.remove(output_file)
