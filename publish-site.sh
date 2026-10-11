#!/bin/sh
# Builds the app, runs the tests, and publishes the site to the gh-pages branch (GitHub Pages):
# the docs (site/) at the root, the app at /app/ with the MNIST files beside it.
set -e
cd "$(dirname "$0")"
npm test
rm -rf _site && mkdir -p _site/app/data
cp -r site/. _site/
cp "Netgraph_ compose neural networks.html" _site/app/index.html
cp -r data/mnist _site/app/data/
touch _site/.nojekyll
url=$(git remote get-url origin)
cd _site
git init -q -b gh-pages
git add -A
git commit -q -m "Site from $(git -C .. rev-parse --short HEAD)"
git push -q -f "$url" gh-pages
cd .. && rm -rf _site
echo "Published to gh-pages."
