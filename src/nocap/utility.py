import networkx as nx
import libsbml

def create_sbml_model_from_nx(dag, output_file='model.xml'):
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("The provided graph is not a Directed Acyclic Graph (DAG)")

    # Create SBMLDocument
    document = libsbml.SBMLDocument(3, 1)
    
    # Create the Model
    model = document.createModel()
    model.setId('grn_model')
    model.setName('Gene Regulatory Network Model')

    # Set global units for time and extent
    model.setTimeUnits('second')
    model.setExtentUnits('mole')
    model.setSubstanceUnits('mole')

    # Define custom unit for per_second
    unit_def = model.createUnitDefinition()
    unit_def.setId('per_second')
    unit = unit_def.createUnit()
    unit.setKind(libsbml.UNIT_KIND_SECOND)
    unit.setExponent(-1)
    unit.setScale(0)
    unit.setMultiplier(1)

    # Create Compartments
    comp = model.createCompartment()
    comp.setId('compartment')
    comp.setConstant(True)
    comp.setSize(1)
    comp.setUnits('litre')

    # Create Species (Nodes as genes or proteins)
    for node in dag.nodes:
        species = model.createSpecies()
        species.setId(node)
        species.setCompartment('compartment')
        species.setInitialAmount(0)
        species.setSubstanceUnits('mole')
        species.setBoundaryCondition(False)
        species.setHasOnlySubstanceUnits(False)
        species.setConstant(False)

    # Create Reactions with regulatory interactions
    for source, target, data in dag.edges(data=True):
        interaction_type = data.get('type', 'activation')  # Default to activation if not specified

        reaction = model.createReaction()
        reaction.setId(f'reaction_{source}_to_{target}')
        reaction.setReversible(False)

        reactant = reaction.createReactant()
        reactant.setSpecies(source)
        reactant.setStoichiometry(1)
        reactant.setConstant(False)

        product = reaction.createProduct()
        product.setSpecies(target)
        product.setStoichiometry(1)
        product.setConstant(False)
        
        # Add regulatory rate law
        kinetic_law = reaction.createKineticLaw()
        parameter = kinetic_law.createParameter()
        parameter.setId(f'k_{source}_to_{target}')
        parameter.setValue(1.0)  # Example rate constant value
        parameter.setUnits('per_second')

        # Ensure formulas fit mole/second
        if interaction_type == 'activation':
            math_ast = libsbml.parseL3Formula(f'k_{source}_to_{target} * {source}')
        elif interaction_type == 'inhibition':
            math_ast = libsbml.parseL3Formula(f'k_{source}_to_{target} / (1 + {source})')
        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")

        kinetic_law.setMath(math_ast)
        kinetic_law.setSubstanceUnits('mole')
        kinetic_law.setTimeUnits('second')

    # Write SBML to file
    libsbml.writeSBMLToFile(document, output_file)

    return document
