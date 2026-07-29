# DOI Archiving Procedure

The repository is configured for Zenodo metadata through `.zenodo.json` and
`CITATION.cff`. A DOI must be issued by Zenodo; it must not be invented or
added before the archive record exists.

## Steps

1. Push this repository to `https://github.com/abbasaqeel-bit/SFO-Path-Planning`.
2. Sign in to Zenodo and open GitHub integration settings.
3. Enable the `SFO-Path-Planning` repository in Zenodo.
4. On GitHub, create a release with a permanent tag, for example `v1.0.0`.
5. Zenodo archives the release and assigns a version-specific DOI and a
   concept DOI.
6. Copy the version-specific DOI into `CITATION.cff` using:

```yaml
identifiers:
  - type: doi
    value: "10.5281/zenodo.XXXXXXX"
```

7. Commit the updated citation file and create a subsequent tagged release.

Use the version-specific DOI in the software citation. The concept DOI may be
used when referring to all released versions of the repository.
