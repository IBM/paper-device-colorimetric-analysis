# paper-device-colorimetric-analysis
 
<!-- This should be the location of the title of the repository, normally the short name -->

<!-- Build Status, is a great thing to have at the top of your repository, it shows that you take your CI/CD as first class citizens -->
<!-- [![Build Status](https://travis-ci.org/jjasghar/ibm-cloud-cli.svg?branch=master)](https://travis-ci.org/jjasghar/ibm-cloud-cli) -->

<!-- Not always needed, but a scope helps the user understand in a short sentance like below, why this repo exists -->
## Scope

This repository relates to the publication with title:

*"A mobile soil analysis system for sustainable agriculture" Ademir Ferreira da Silva el al.*

Available at: TBD

and the corresponding dataset repository archived at: 

https://doi.org/10.24435/materialscloud:56-6p

<!-- A more detailed Usage or detailed explaination of the repository here -->
## Usage

This repository comprises the following Jupyter Notebooks for the analysis and model training with colorimetric data extracted from chemical reactions on paper-based sensing devices.

In sequencial order of application:

#### *Calibration Feature Extraction.ipynb*

This notebook extract the colorimetric information from the images captured of paper devices, and saving the RGB values in a csv file.
This notebook uses the images in the repository: https://doi.org/10.24435/materialscloud:56-6p

#### *Colorimetric Model Training.ipynb*

This notebook loads the csv file with RGB data per paper device, after adding a column with the 'Class' of that data based on the pH value of the sample applied to the paper device, and trains two openCV Logistic Regression models that are saved into XML files for application with a mobile device.  
This notebook uses the csv files with RGB data collected for two pH indicators and available at https://doi.org/10.24435/materialscloud:56-6p

#### *AgroPad Analysis Demo.ipynb*

This notebook shows how each image of a paper device captured outdoors with the mobile device is processed to compensate for illumination differences with the calibrationd dataset collected under laboratory illumination conditions. This notebook also goes through the subsequent steps of importing and then applying the trained logistic regression models to the newly captured and treated color data. Reference logistic regression models can be found under the folder '\ML_models' and illumination references used by the notebook can be found under folder '\Illumination_references'

#### *uIPL_2022_v2.py*

Library of functions used by the above notebooks. 


-------------------------------------


* [LICENSE](LICENSE)
* [README.md](README.md)
* [CONTRIBUTING.md](CONTRIBUTING.md)
* [MAINTAINERS.md](MAINTAINERS.md)
<!-- A Changelog allows you to track major changes and things that happen, https://github.com/github-changelog-generator/github-changelog-generator can help automate the process 
* [CHANGELOG.md](CHANGELOG.md) -->

<!-- > These are optional -->

<!-- The following are OPTIONAL, but strongly suggested to have in your repository. 
* [dco.yml](.github/dco.yml) - This enables DCO bot for you, please take a look https://github.com/probot/dco for more details.
* [travis.yml](.travis.yml) - This is a example `.travis.yml`, please take a look https://docs.travis-ci.com/user/tutorial/ for more details. -->

<!-- These may be copied into a new or existing project to make it easier for developers not on a project team to collaborate. -->

<!-- A notes section is useful for anything that isn't covered in the Usage or Scope. Like what we have below. 
## Notes
**NOTE: While this boilerplate project uses the Apache 2.0 license, when
establishing a new repo using this template, please use the
license that was approved for your project.**
**NOTE: This repository has been configured with the [DCO bot](https://github.com/probot/dco).
When you set up a new repository that uses the Apache license, you should
use the DCO to manage contributions. The DCO bot will help enforce that.
Please contact one of the IBM GH Org stewards.** -->

<!-- Questions can be useful but optional, this gives you a place to say, "This is how to contact this project maintainers or create PRs -->
<!-- If you have any questions or issues you can create a new [issue here][issues].
Pull requests are very welcome! Make sure your patches are well tested.
Ideally create a topic branch for every separate change you make. For
example:
1. Fork the repo
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Added some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request -->

## License

<!-- All source files must include a Copyright and License header. The SPDX license header is 
preferred because it can be easily scanned. -->

If you would like to see the detailed LICENSE click [here](LICENSE).

```text
#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: BSD-3-Clause
#
```
## Authors

- Matheus Esteves Ferreira 
- Jaione Tirapu Azpiroz 

[issues]: https://github.com/IBM/repo-template/issues/new
