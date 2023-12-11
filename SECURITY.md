# Security Policy

## Supported Versions

We aim to update all Annif dependencies to their latest versions
on each Annif minor version release, but this can be restricted by the
[backward compatibility policy](https://github.com/NatLibFi/Annif/wiki/Backward-compatibility-between-Annif-releases).

However, the [dependencies of a given Annif release](https://github.com/NatLibFi/Annif/blob/main/pyproject.toml)
are pinned only on minor version level, so all patch level fixes of dependencies
can be applied to an Annif installation just by running
`pip install annif --upgrade`.

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

Please include the following information along with your report:

- A descriptive title, clearly stating the nature and object software (Annif) of the report.
- Your name and affiliation (if any).
- A description of the technical details of the vulnerabilities.
- A minimal example of the vulnerability.
- An explanation of who can exploit this vulnerability, and what they gain when doing so.
- Whether this vulnerability is public or known to third parties. If it is, please provide details.
