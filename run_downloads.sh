#!/bin/bash

FAILED=()
LOGFILE="download_log.txt"
> "$LOGFILE"

echo "Starting downloads..."

while read -r line; do
  echo "Running: $line" | tee -a "$LOGFILE"
  if eval "$line" >>"$LOGFILE" 2>&1; then
    echo "✅ Success" | tee -a "$LOGFILE"
  else
    echo "❌ Failed: $line" | tee -a "$LOGFILE"
    FAILED+=("$line")
  fi
  echo "" >> "$LOGFILE"
done <<EOF
pip download --no-deps abcvoting==2.10.0 -d cc_libs
pip download --no-deps absl-py==2.1.0 -d cc_libs
pip download --no-deps appnope==0.1.4 -d cc_libs
pip download --no-deps asttokens==2.4.1 -d cc_libs
pip download --no-deps astunparse==1.6.3 -d cc_libs
pip download --no-deps attrs==24.2.0 -d cc_libs
pip download --no-deps backcall==0.2.0 -d cc_libs
pip download --no-deps beautifulsoup4==4.12.3 -d cc_libs
pip download --no-deps bleach==6.1.0 -d cc_libs
pip download --no-deps certifi==2024.6.2 -d cc_libs
pip download --no-deps cffi==1.15.1 -d cc_libs
pip download --no-deps charset-normalizer==3.3.2 -d cc_libs
pip download --no-deps contourpy==1.2.1 -d cc_libs
pip download --no-deps cycler==0.12.1 -d cc_libs
pip download --no-deps decorator==5.1.1 -d cc_libs
pip download --no-deps defusedxml==0.7.1 -d cc_libs
pip download --no-deps Deprecated==1.2.14 -d cc_libs
pip download --no-deps docopt==0.6.2 -d cc_libs
pip download --no-deps executing==2.1.0 -d cc_libs
pip download --no-deps fastjsonschema==2.20.0 -d cc_libs
pip download --no-deps filelock==3.14.0 -d cc_libs
pip download --no-deps flatbuffers==24.3.25 -d cc_libs
pip download --no-deps fonttools==4.53.0 -d cc_libs
pip download --no-deps fsspec==2024.6.0 -d cc_libs
pip download --no-deps gast==0.5.4 -d cc_libs
pip download --no-deps google-pasta==0.2.0 -d cc_libs
pip download --no-deps grpcio==1.64.1 -d cc_libs
pip download --no-deps h5py==3.11.0 -d cc_libs
pip download --no-deps idna==3.7 -d cc_libs
pip download --no-deps immutabledict==4.2.0 -d cc_libs
pip download --no-deps ipython==8.12.3 -d cc_libs
pip download --no-deps jedi==0.19.1 -d cc_libs
pip download --no-deps Jinja2==3.1.4 -d cc_libs
pip download --no-deps joblib==1.4.2 -d cc_libs
pip download --no-deps jsonschema==4.23.0 -d cc_libs
pip download --no-deps jsonschema-specifications==2023.12.1 -d cc_libs
pip download --no-deps jupyter_client==8.6.2 -d cc_libs
pip download --no-deps jupyter_core==5.7.2 -d cc_libs
pip download --no-deps jupyterlab_pygments==0.3.0 -d cc_libs
pip download --no-deps kiwisolver==1.4.5 -d cc_libs
pip download --no-deps libclang==18.1.1 -d cc_libs
pip download --no-deps llvmlite==0.41.1 -d cc_libs
pip download --no-deps Markdown==3.6 -d cc_libs
pip download --no-deps markdown-it-py==3.0.0 -d cc_libs
pip download --no-deps MarkupSafe==2.1.5 -d cc_libs
pip download --no-deps matplotlib==3.9.0 -d cc_libs
pip download --no-deps matplotlib-inline==0.1.7 -d cc_libs
pip download --no-deps mdurl==0.1.2 -d cc_libs
pip download --no-deps mip==1.15.0 -d cc_libs
pip download --no-deps mistune==3.0.2 -d cc_libs
pip download --no-deps ml-dtypes==0.3.2 -d cc_libs
pip download --no-deps mpmath==1.3.0 -d cc_libs
pip download --no-deps namex==0.0.8 -d cc_libs
pip download --no-deps nashpy==0.0.40 -d cc_libs
pip download --no-deps nbclient==0.10.0 -d cc_libs
pip download --no-deps nbconvert==7.16.4 -d cc_libs
pip download --no-deps nbformat==5.10.4 -d cc_libs
pip download --no-deps networkx==3.3 -d cc_libs
pip download --no-deps numba==0.58.1 -d cc_libs
pip download --no-deps numpy==1.26.4 -d cc_libs
pip download --no-deps opt-einsum==3.3.0 -d cc_libs
pip download --no-deps optree==0.11.0 -d cc_libs
pip download --no-deps ortools==9.10.4067 -d cc_libs
pip download --no-deps packaging==24.0 -d cc_libs
pip download --no-deps pandas==2.2.2 -d cc_libs
pip download --no-deps pandocfilters==1.5.1 -d cc_libs
pip download --no-deps parso==0.8.4 -d cc_libs
pip download --no-deps pexpect==4.9.0 -d cc_libs
pip download --no-deps pickleshare==0.7.5 -d cc_libs
pip download --no-deps pillow==10.3.0 -d cc_libs
pip download --no-deps pipdeptree==2.23.1 -d cc_libs
pip download --no-deps pipreqs==0.5.0 -d cc_libs
pip download --no-deps platformdirs==4.3.3 -d cc_libs
pip download --no-deps pref-voting==1.13.18 -d cc_libs
pip download --no-deps preflibtools==2.0.22 -d cc_libs
pip download --no-deps prefsampling==0.1.20 -d cc_libs
pip download --no-deps prompt_toolkit==3.0.47 -d cc_libs
pip download --no-deps protobuf==4.25.3 -d cc_libs
pip download --no-deps ptyprocess==0.7.0 -d cc_libs
pip download --no-deps pure_eval==0.2.3 -d cc_libs
pip download --no-deps pycparser==2.22 -d cc_libs
pip download --no-deps Pygments==2.18.0 -d cc_libs
pip download --no-deps pyparsing==3.1.2 -d cc_libs
pip download --no-deps pyrankvote==2.0.6 -d cc_libs
pip download --no-deps python-dateutil==2.9.0.post0 -d cc_libs
pip download --no-deps pytorch-ignite -d cc_libs
pip download --no-deps pytz==2024.1 -d cc_libs
pip download --no-deps pyzmq==26.2.0 -d cc_libs
pip download --no-deps random2==1.0.2 -d cc_libs
pip download --no-deps referencing==0.35.1 -d cc_libs
pip download --no-deps requests==2.32.3 -d cc_libs
pip download --no-deps rich==13.7.1 -d cc_libs
pip download --no-deps rpds-py==0.20.0 -d cc_libs
pip download --no-deps ruamel.yaml==0.18.6 -d cc_libs
pip download --no-deps ruamel.yaml.clib==0.2.8 -d cc_libs
pip download --no-deps scikit-learn==1.5.0 -d cc_libs
pip download --no-deps scipy==1.13.1 -d cc_libs
pip download --no-deps seaborn==0.13.2 -d cc_libs
pip download --no-deps six==1.16.0 -d cc_libs
pip download --no-deps snakeviz==2.2.0 -d cc_libs
pip download --no-deps soupsieve==2.6 -d cc_libs
pip download --no-deps stack-data==0.6.3 -d cc_libs
pip download --no-deps sympy==1.12.1 -d cc_libs
pip download --no-deps tabulate==0.9.0 -d cc_libs
pip download --no-deps termcolor==2.4.0 -d cc_libs
pip download --no-deps threadpoolctl==3.5.0 -d cc_libs
pip download --no-deps tinycss2==1.3.0 -d cc_libs
pip download --no-deps tornado==6.4.1 -d cc_libs
pip download --no-deps traitlets==5.14.3 -d cc_libs
pip download --no-deps typing_extensions==4.12.1 -d cc_libs
pip download --no-deps tzdata==2024.1 -d cc_libs
pip download --no-deps urllib3==2.2.1 -d cc_libs
pip download --no-deps wcwidth==0.2.13 -d cc_libs
pip download --no-deps webencodings==0.5.1 -d cc_libs
pip download --no-deps Werkzeug==3.0.3 -d cc_libs
pip download --no-deps wrapt==1.16.0 -d cc_libs
pip download --no-deps yarg==0.1.9 -d cc_libs
EOF

echo ""
echo "=== FAILED DOWNLOADS ==="
for f in "${FAILED[@]}"; do
  echo "$f"
done

# Save failed ones to file
echo "Saving failed downloads to failed_downloads.txt..."
printf "%s\n" "${FAILED[@]}" > failed_downloads.txt