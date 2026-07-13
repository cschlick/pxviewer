import molviewspec as mvs


def create_example_view(url: str) -> str:
    """Build an example MVS scene from a structure URL."""
    builder = mvs.create_builder()
    structure = builder.download(url=url).parse(format="bcif").model_structure()
    structure.component(selector="polymer").representation(type="cartoon").color(color="green")
    structure.component(selector="ligand").representation(type="ball_and_stick").color(color="#cc3399")
    return builder.get_state().model_dump_json()
