## Environments

Environment files and setup notes.

### Contents

- conda/mamba environment YAML files, `renv` files, module lists, or other setup files.
- Instructions for recreating the software environment.

### Runtime environments

Define default runtimes by language once, then document only scripts that need a non-default runtime. Agents should use these defaults instead of creating their own environments.

Edit these defaults for each project:

| Language | Default environment | Setup command | Run command pattern | Interactive command |
| --- | --- | --- | --- | --- |
| R | `<r-env-name>` or `envs/<r-env>.yml` | `mamba env create -f envs/<r-env>.yml` | `mamba run -n <r-env-name> Rscript <script> <args>` | `mamba run -n <r-env-name> R` |
| Python | `<py-env-name>` or `envs/<py-env>.yml` | `mamba env create -f envs/<py-env>.yml` | `mamba run -n <py-env-name> python <script> <args>` | `mamba run -n <py-env-name> python` |
| Bash | System shell or project module setup | Document any required modules here | `bash <script> <args>` | `bash` |
| Quarto | R or Python default, as appropriate | Use the matching language setup | `quarto render <file>` | `quarto preview <file>` |

### Script-specific overrides

Only add rows here for scripts that do not use the language default.

| Script path | Runtime | Environment override | Setup command | Run command | Interactive/check command |
| --- | --- | --- | --- | --- | --- |
| `scripts/example.R` | R | `<special-r-env>` | `mamba env create -f envs/<special-r-env>.yml` | `mamba run -n <special-r-env> Rscript scripts/example.R` | `mamba run -n <special-r-env> R` |

If no default or override exists for a script, document the missing runtime instead of guessing.
