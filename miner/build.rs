use vergen_gitcl::{Emitter, Gitcl};

fn main() -> anyhow::Result<()> {
    let git = Gitcl::builder().sha(true).build();
    Emitter::default().default_on_error().add_instructions(&git)?.emit()
}
