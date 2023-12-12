# Security Policy

## Supported Versions

The [most recent Annif (major/minor) release](https://github.com/NatLibFi/Annif/releases)
is considered supported,
in the sense that if a serious bug or vulnerability is encountered in it,
a patch release is made to fix the issue.

Generally, we aim to update all dependencies to their latest versions
on each Annif major/minor release, but this can be restricted by the
[backward compatibility policy](https://github.com/NatLibFi/Annif/wiki/Backward-compatibility-between-Annif-releases).
However, note that the [dependencies of a given Annif release](https://github.com/NatLibFi/Annif/blob/main/pyproject.toml)
are pinned only on minor version level, so all patch level fixes of dependencies
can be applied to an Annif installation
(either manually updating the outdated packages or recreating the virtual environment and reinstalling Annif).

### Docker image
The Docker image of the latest Annif release in the
[quay.io repository](https://quay.io/repository/natlibfi/annif?tab=tags)
is rebuilt from time to time in order to update both system packages and Annif dependencies in the image.

The security scanner that is used on quay.io is
[Clair](https://access.redhat.com/documentation/en-us/red_hat_quay/3/html/about_quay_io/clair-vulnerability-scanner).
Typically the scanner detects many vulnerabilities at any moment in the Annif image, even several tens.
However, there exist patches for only some of the vulnerabilities,
and due to the way that Annif uses the dependencies, most of the detected vulnerabilities
do not apply to Annif use.

## Reporting a Vulnerability

Thank you for improving the security of Annif.
We value your findings, and we'd be grateful if you report
any concerns or vulnerabilities directly to `finto-posti@helsinki.fi`.
Note that Annif team is a part of the larger Finto team,
which has resources for the contact service throughout the year.

If the security vulnerability is in a third-party software library,
please report it also to the team maintaining it.

Each security concern will be assigned to a handler from our team,
who will contact you if there's a need for additional information.
We confirm the problem and keep you informed of the fix.

Make sure to add the following details when submitting your report:

- A clear and descriptive title that outlines the report's subject and the software it pertains to (Annif).
- Break down the technical aspects of the vulnerability in your description.
- A minimal example showcasing the vulnerability.
- An explanation who has the potential to exploit this vulnerability and the benefits they would derive from doing so.
- Whether the vulnerability is public knowledge or known to third parties, and if so, share relevant details.
