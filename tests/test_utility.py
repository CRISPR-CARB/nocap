from nocap.utility import create_sbml_model_from_nx

import networkx as nx
import libsbml
import os


def test_create_sbml_model_from_nx():

    def test_valid_sbml(document):
        errors = document.checkConsistency()
        if errors > 0:
            print("---- SBML Errors ----")
            for i in range(errors):
                print(document.getError(i).getMessage())
            print("---------------------")
        writer = libsbml.SBMLWriter()
        sbml_str = writer.writeToString(document)
        print("---- SBML Document ----")
        print(sbml_str)
        print("------------------------")
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
        for reaction_id, equation in expected_reactions.items():
            reaction = model.getReaction(reaction_id)
            assert reaction is not None, f"Reaction {reaction_id} not found"
            math_ast = reaction.getKineticLaw().getMath()
            formula = libsbml.formulaToString(math_ast)
            assert formula == equation, f"Equation for {reaction_id} does not match: {formula} != {equation}"

    # Test for correct parameter values (i.e. rate constants)
    def test_parameters(document, expected_parameters):
        model = document.getModel()
        for reaction in model.getListOfReactions():
            kinetic_law = reaction.getKineticLaw()
            for param in kinetic_law.getListOfParameters():
                param_id = param.getId()
                value = param.getValue()
                assert param_id in expected_parameters, f"Parameter {param_id} not expected"
                assert value == expected_parameters[param_id], f"Value for {param_id} does not match: {value} != {expected_parameters[param_id]}"

    # Test for species concentrations test
    def test_species_concentrations(document, expected_concentrations):
        model = document.getModel()
        for species_id, expected_concentration in expected_concentrations.items():
            species = model.getSpecies(species_id)
            assert species is not None, f"Species {species_id} not found"
            init_amount = species.getInitialAmount()
            assert init_amount == expected_concentration, f"Concentration for {species_id} does not match: {init_amount} != {expected_concentration}"


    # Create a sample GRN
    dag = nx.DiGraph()
    dag.add_edge('GeneA', 'ProteinB', type='activation')
    dag.add_edge('ProteinB', 'GeneC', type='inhibition')

    # Generate SBML model and save to file
    document = create_sbml_model_from_nx(dag, output_file='test_model.xml')

    # Expected values for testing
    expected_species = ['GeneA', 'ProteinB', 'GeneC']
    expected_reactions = {
        'reaction_GeneA_to_ProteinB': 'k_GeneA_to_ProteinB * GeneA',
        'reaction_ProteinB_to_GeneC': 'k_ProteinB_to_GeneC / (1 + ProteinB)'
    }
    expected_parameters = {
        'k_GeneA_to_ProteinB': 1.0,
        'k_ProteinB_to_GeneC': 1.0
    }
    expected_concentrations = {
        'GeneA': 0.0,
        'ProteinB': 0.0,
        'GeneC': 0.0
    }

    # Run tests
    test_valid_sbml(document)
    test_species(document, expected_species)
    test_reactions(document, expected_reactions.keys())
    test_compartment(document, 'compartment')
    test_reaction_equations(document, expected_reactions)
    test_parameters(document, expected_parameters)
    test_species_concentrations(document, expected_concentrations)

    # Cleanup
    os.remove('test_model.xml')
