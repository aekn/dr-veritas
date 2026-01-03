PMC's article datasets are freely accessible on Amazon Web Services.

The files come in NISO JATS XML format as well as plain text extracted from the XML.

Explore the S3 bucket:
```bash
aws --no-sign-request s3 ls s3://pmc-oa-opendata
```

Download a single file by its PMCID:
```bash
# with PMCID like PM1234567
aws --no-sign-request s3 cp s3://pmc-oa-opendata/oa_comm/xml/all/{PMCID}.xml .
```

Download subsets of files utilizing various options:
```bash
aws --no-sign-request s3 cp s3://pmc-oa-opendata ./pmc-test/ --exclude "*" --include "oa_comm/xml/all/" --recursive
```

Resources:
- [\[PMC Dataset Using AWS\]](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/)
- [\[AWS S3 CLI Reference\]](https://docs.aws.amazon.com/cli/latest/reference/s3/)
