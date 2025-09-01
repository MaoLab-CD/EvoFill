#!/usr/bin/env bash
# set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 input.vcf.gz"
  exit 1
fi

in=$1
prefix=${in%%.vcf.gz}

# 1. 抽取样本名
bcftools query -l "$in" > samples.txt

# 2. 随机 7:3 切分
shuf samples.txt > samples.shuf.txt
split -n l/10 -d samples.shuf.txt chunk_
cat chunk_0{0..6} > train.list
cat chunk_0{7..9} > val.list
rm chunk_*

# 3. 输出子 VCF
bcftools view -S train.list -Oz -o "${prefix}_train.vcf.gz" "$in"
bcftools view -S val.list  -Oz -o "${prefix}_val.vcf.gz"  "$in"

echo "Done. Output files:"
echo "  ${prefix}_train.vcf.gz (${prefix}_train.vcf.gz.tbi)"
echo "  ${prefix}_val.vcf.gz  (${prefix}_val.vcf.gz.tbi)"