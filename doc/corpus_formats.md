# Corpus formats

Annif uses different kinds of corpora. This document specifies the formats.

## Full-text document corpus

The full text corpus is a directory with UTF-8 encoded text files that have
the file extension `.txt`.

The directory may also contain **subject files** that list the assigned
subjects for each file. The file name is the same as the document file, but
with the file extension `.key`. For example, `document1.txt` may have a
corresponding subject file `document1.key`. Subject files come in two
formats:

### Simple subject file format

This file lists subject labels, UTF-8 encoded, one per line. For example:

```
networking
computer science
Internet Protocol
```

Note that the labels must exactly match the preferred labels of concepts in
the subject vocabulary.

This format corresponds to the [Maui topic file
format](https://code.google.com/archive/p/maui-indexer/wikis/Usage.wiki).

### Extended subject file format

This is otherwise similar to the simple subject file format, but the `.key`
file is now a UTF-8 encoded TSV (tab separated values) file where the first
column contains a subject URI and the second column its label. For example:

```
<http://example.org/thesaurus/subj1>	networking
<http://example.org/thesaurus/subj2>	computer science
<http://example.org/thesaurus/subj3>	Internet Protocol
```

Any additional columns beyond the first two are ignored.

When using this format, subject comparison is performed based on URIs, not
the labels. Since URIs are more persistent than labels, this ensures that
subjects can be matched even if the labels have changed in the subject
vocabulary.

## Subjet corpus

TBD

## Metadata only corpus

TBD
