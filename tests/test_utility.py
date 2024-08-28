import networkx as nx
import libsbml
import os
from nocap.utility import create_sbml_model_from_nx
import tellurium as te
import re

def test_create_sbml_model_from_nx():

    ### Sub tests
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
        assert compartments[0].getId() == expected_compartment_id, "Compartment ID does not match"

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


    def normalize_ode(ode):
        """Helper function to normalize ODE strings for comparison."""
        return ode.strip('()+ ').replace(' ', '').replace('\t', '')

    def test_odes(document, expected_odes):
        # Convert SBML document back to Antimony to inspect ODEs
        sbml_str = libsbml.writeSBMLToString(document)
        r = te.loadSBMLModel(sbml_str)

        # Get the ODEs from the model
        odes = te.getODEsFromModel(r)
        print(f'odes from rr model: {odes}')

        # Make sure we access reactions correctly from the SBML document
        sbml_document = libsbml.readSBMLFromString(sbml_str)
        model = sbml_document.getModel()

        # Extract reaction formulas
        reactions = {reaction.getId(): reaction.getKineticLaw().getFormula() for reaction in model.getListOfReactions()}
        print(f'reactions: {reactions}')

        # Helper function to extract the ODEs as a dictionary
        def extract_odes_as_dict(ode_text):
            ode_dict = {}
            for line in ode_text.split("\n"):
                if " = " in line and "d" in line:
                    try:
                        species_id, ode = line.split(' = ', 1)
                        species_id = species_id.split('d')[1].split('/')[0]  # Extract species ID from dGene/dt
                        ode_dict[species_id] = ode.strip()  # Remove any leading/trailing whitespace
                    except ValueError:
                        continue  # Skip lines that don't fit the expected pattern
            return ode_dict

        # Extract ODEs as a dictionary
        actual_odes_dict = extract_odes_as_dict(odes)
        print(f'actual odes dict: {actual_odes_dict}')

        # Replace fluxes with corresponding formulas and clean up notation
        pow_to_exp = re.compile(r'pow\(([^,]+), ([^)]+)\)')
        for species_id, ode in actual_odes_dict.items():
            for flux, formula in reactions.items():
                # Handle the 'v' prefix
                flux_with_v = 'v' + flux
                actual_odes_dict[species_id] = actual_odes_dict[species_id].replace(flux_with_v, formula)
            
            # Replace pow() notation with ^ notation
            actual_odes_dict[species_id] = pow_to_exp.sub(r'\1^\2', actual_odes_dict[species_id])
            
            # Ensure no leading '+ ' in actual equations if present
            actual_odes_dict[species_id] = actual_odes_dict[species_id].lstrip('+ ')

        print(f'actual odes dict (with formulas): {actual_odes_dict}')

        # Loop through expected ODEs and assert they match the generated ones
        for species_id, expected_ode in expected_odes.items():
            print(f'species id: {species_id}')
            print(f'expected ode: {expected_ode}')
            if species_id in actual_odes_dict:
                actual_ode = actual_odes_dict[species_id]
                normalized_expected_ode = normalize_ode(expected_ode)
                normalized_actual_ode = normalize_ode(actual_ode)
                print(f"Expected ODE for {species_id}: {normalized_expected_ode}")
                print(f"Actual   ODE for {species_id}: {normalized_actual_ode}")
                assert normalized_expected_ode == normalized_actual_ode, f"ODE for {species_id} does not match: Expected {normalized_expected_ode}, but found {normalized_actual_ode}."
            else:
                # Ensure that constants without ODEs are explicitly tested for constancy
                if expected_ode == '0':
                    print(f"Ensuring {species_id} is constant.")
                    assert model.getSpecies(species_id).getBoundaryCondition(), f"Species {species_id} is not set as constant (boundary condition!=True)."
                else:
                    print(f"Expected an ODE for {species_id}, but none was found.")
                    assert False, f"Expected ODE for {species_id} but found none."

        print("All ODEs match expected values.")




    # Additional Tests for Error Handling
    def test_bad_node_values():
        dag_invalid1 = nx.DiGraph()
        dag_invalid1.add_node('GeneA', value='invalid')
        dag_invalid1.add_node('GeneB', value=0.0)
        dag_invalid1.add_edge('GeneA', 'GeneB', interaction_type='activation')
        output_file = 'test_invalid_model1.xml'
        try:
            create_sbml_model_from_nx(dag_invalid1, output_file)
        except ValueError as e:
            assert str(e) == "Node GeneA has an invalid 'value' attribute: invalid. Must be a non-negative number."

        dag_invalid2 = nx.DiGraph()
        dag_invalid2.add_node('GeneA', value=1.0)
        dag_invalid2.add_node('GeneB', value=-1.0)
        dag_invalid2.add_edge('GeneA', 'GeneB', interaction_type='activation')
        output_file = 'test_invalid_model2.xml'
        try:
            create_sbml_model_from_nx(dag_invalid2, output_file)
        except ValueError as e:
            assert str(e) == "Node GeneB has an invalid 'value' attribute: -1.0. Must be a non-negative number."

    def test_non_dag_input():
        dag_invalid = nx.DiGraph()  # Changing this to DiGraph to make it directed but will create a cycle
        dag_invalid.add_node('GeneA', value=1.0)
        dag_invalid.add_node('GeneB', value=0.0)
        dag_invalid.add_edge('GeneA', 'GeneB', interaction_type='activation')
        dag_invalid.add_edge('GeneB', 'GeneA', interaction_type='repression')  # Creates a cycle
        output_file = 'test_invalid_model.xml'
        try:
            create_sbml_model_from_nx(dag_invalid, output_file)
        except ValueError as e:
            assert str(e) == "Input graph must be a directed acyclic graph (DAG)."

    def test_bad_interaction_type():
        dag_invalid = nx.DiGraph()
        dag_invalid.add_node('GeneA', value=1.0)
        dag_invalid.add_node('GeneB', value=0.0)
        dag_invalid.add_edge('GeneA', 'GeneB', interaction_type='invalid')
        output_file = 'test_invalid_model.xml'
        try:
            create_sbml_model_from_nx(dag_invalid, output_file)
        except ValueError as e:
            assert str(e) == "Unknown interaction type: invalid"


    ### SETUP
    # Valid DAGs with correct node values and interaction types
    # Example DAG A to B
    dagA_to_B = nx.DiGraph()
    dagA_to_B.add_node('GeneA', value=1.0)
    dagA_to_B.add_node('GeneB', value=0.0)
    dagA_to_B.add_edge('GeneA', 'GeneB', interaction_type='activation')

    # Example DAG A to B and A to C
    dagA_to_B_A_to_C = nx.DiGraph()
    dagA_to_B_A_to_C.add_node('GeneA', value=1.0)
    dagA_to_B_A_to_C.add_node('GeneB', value=0.0)
    dagA_to_B_A_to_C.add_node('GeneC', value=0.0)
    dagA_to_B_A_to_C.add_edge('GeneA', 'GeneB', interaction_type='activation')
    dagA_to_B_A_to_C.add_edge('GeneA', 'GeneC', interaction_type='repression')

    test_cases = [
        (
            dagA_to_B,
            ['GeneA', 'GeneB'],
            {
                'J_GeneA_to_GeneB': 'beta_GeneA_to_GeneB * GeneA^n_GeneA_to_GeneB / (K_GeneA_to_GeneB^n_GeneA_to_GeneB + GeneA^n_GeneA_to_GeneB)'
            },
            {
                'beta_GeneA_to_GeneB': 1.0,
                'K_GeneA_to_GeneB': 1.0,
                'n_GeneA_to_GeneB': 1.0
            },
            {
                'GeneA': 1.0,
                'GeneB': 0.0
            },
            # Expected ODEs
            {
                'GeneA': '0',
                'GeneB': ' + (beta_GeneA_to_GeneB * GeneA^n_GeneA_to_GeneB / (K_GeneA_to_GeneB^n_GeneA_to_GeneB + GeneA^n_GeneA_to_GeneB))'
            }
        ),
        (
            dagA_to_B_A_to_C,
            ['GeneA', 'GeneB', 'GeneC'],
            {
                'J_GeneA_to_GeneB': 'beta_GeneA_to_GeneB * GeneA^n_GeneA_to_GeneB / (K_GeneA_to_GeneB^n_GeneA_to_GeneB + GeneA^n_GeneA_to_GeneB)',
                'J_GeneA_to_GeneC': 'beta_GeneA_to_GeneC / (1 + GeneA^n_GeneA_to_GeneC / K_GeneA_to_GeneC^n_GeneA_to_GeneC)'  # Note the repression formula here
            },
            {
                'beta_GeneA_to_GeneB': 1.0,
                'K_GeneA_to_GeneB': 1.0,
                'n_GeneA_to_GeneB': 1.0,
                'beta_GeneA_to_GeneC': 1.0,
                'K_GeneA_to_GeneC': 1.0,
                'n_GeneA_to_GeneC': 1.0
            },
            {
                'GeneA': 1.0,
                'GeneB': 0.0,
                'GeneC': 0.0
            },
            # Expected ODEs
            {
                'GeneA': '0',
                'GeneB': ' + (beta_GeneA_to_GeneB * GeneA^n_GeneA_to_GeneB / (K_GeneA_to_GeneB^n_GeneA_to_GeneB + GeneA^n_GeneA_to_GeneB))',
                'GeneC': ' + (beta_GeneA_to_GeneC / (1 + GeneA^n_GeneA_to_GeneC / K_GeneA_to_GeneC^n_GeneA_to_GeneC))'
            }
        )
    ]



    ### RUN TESTS
    # Running tests for error handling
    test_bad_node_values()
    test_non_dag_input()
    test_bad_interaction_type()

    for dag, expected_species, expected_reactions, expected_parameters, expected_concentrations, expected_odes in test_cases:
        # Define the output file path
        output_file = 'test_model.xml'

        # Generate SBML model
        create_sbml_model_from_nx(dag, output_file)

        # Read SBML document from the file using libsbml
        document = libsbml.readSBML(output_file)

        # Running the sub-tests
        #test_valid_sbml(document)
        test_species(document, expected_species)
        test_reactions(document, expected_reactions.keys())
        test_compartment(document, 'default_compartment')
        test_reaction_equations(document, expected_reactions)
        test_parameters(document, expected_parameters)
        test_species_concentrations(document, expected_concentrations)
        test_odes(document, expected_odes)

        # Cleanup the generated file
        os.remove(output_file)

        