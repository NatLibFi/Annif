# Security Policy

## Supported Versions

The [most recent Annif major/minor release](https://github.com/NatLibFi/Annif/releases)
is considered supported,
in the sense that if a serious bug or vulnerability is encountered in it,
we relase a patch to fix the issue.

Generally, we aim to update all dependencies to their latest versions on each Annif major/minor release.
However, note that most of the [dependencies of a given Annif release](https://github.com/NatLibFi/Annif/blob/main/pyproject.toml)
are pinned only on minor version level, so patch level fixes of (most) dependencies can be applied to an Annif installation,
by either manually updating the outdated packages or recreating the virtual environment from scratch and reinstalling Annif.

### Docker image
We rebuild and publish a new Docker image of the latest Annif release in the
[quay.io repository](https://quay.io/repository/natlibfi/annif?tab=info)
when it is considered necessary in order to update both system packages and Annif dependencies of the image.
A new image is published about once every month.

The security scanner that is used on quay.io is
[Clair](https://access.redhat.com/documentation/en-us/red_hat_quay/3/html/about_quay_io/clair-vulnerability-scanner).
You can see the vulnerabilities detected in an image by navigating via the link in the Security Scan column of the [tags view](https://quay.io/repository/natlibfi/annif?tab=tags),
see the screenshot below.

The scanner typically detects many vulnerabilities, that is several tens, in the packages of the images, even when they have been rebuild recently.
However, there exist patches for only some of the vulnerabilities,
and due to the way that Annif uses the dependencies, most of the detected vulnerabilities
do not apply to Annif use.

<img src="https://github.com/NatLibFi/Annif/assets/34240031/bab1316e-57fb-46a4-8ec0-94a414b26e2a" width="500">

## Reporting a Vulnerability

We value your findings, and we would be grateful if you report
any concerns or vulnerabilities by email to **`finto-posti@helsinki.fi`**.
Note that Annif team is a part of the larger Finto team,
which has resources for the contact service throughout the year.

If the security vulnerability is in a third-party software library,
please report it also to the team maintaining it.

Each security concern will be assigned to a handler from our team,
who will contact you if there is a need for additional information.
We confirm the problem and keep you informed of the fix.

To facilitate a quick and accurate response make sure to include the following details when submitting your report:

- A clear and descriptive title that outlines the report's subject and the software it pertains to (Annif).
- The versions of Annif, its dependencies and the (possible) other related software that give rise to the vulnerability.
- Break down the technical aspects of the vulnerability in your description.
- A minimal example showcasing the vulnerability.
- An explanation who has the potential to exploit this vulnerability and the benefits they would derive from doing so.
- Whether the vulnerability is public knowledge or known to third parties, and if so, share relevant details.
