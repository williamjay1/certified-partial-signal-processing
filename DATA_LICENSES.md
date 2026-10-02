# Licenses and attribution

The root MIT license covers original study software and its software documentation authored for this project. It does not replace third-party licenses and does not grant a new license for upstream dataset fields.

## Original research outputs

The model-generated probabilities, certificate values, timing observations, aggregate analyses, original scientific artwork, and fitted study model artifacts are made available by Junjie Zhang under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Cite this repository and the relevant source datasets. Where a file also contains upstream row indices, subject identifiers, activity/condition labels, or split identifiers, the upstream fields retain the source-specific terms below. This is a mixed-license research package, not a blanket relicensing of the UCI datasets.

## Smartphone activity recognition

J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto, X. Parra, *Human Activity Recognition Using Smartphones*, UCI Machine Learning Repository (2013), [DOI 10.24432/C54S4K](https://doi.org/10.24432/C54S4K). Cite D. Anguita et al., *A Public Domain Dataset for Human Activity Recognition Using Smartphones*, ESANN 2013, as requested by the original dataset README.

The [current UCI page](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) states CC BY 4.0. The original downloaded archive's README additionally prohibits commercial use and requests acknowledgment. These conflicting upstream statements are reported explicitly; we do not remove the original restriction or reinterpret it as permission to relicense all source material. Raw inertial signals and source feature caches are not redistributed here. Retrieve them from UCI and retain the upstream README. Saved model outcomes include released pseudonymous subject IDs, labels, and row/split alignment fields for research verification; those fields are not attributed as original work by Junjie Zhang and remain subject to upstream terms. For commercial reuse of those fields, obtain clarification from the dataset rights holders. The conflict does not prohibit legitimate academic verification of this study.

## Hydraulic monitoring

N. Helwig, E. Pignanelli, A. Schütze, *Condition monitoring of hydraulic systems*, UCI Machine Learning Repository (2015), [DOI 10.24432/C5CW21](https://doi.org/10.24432/C5CW21). The [current UCI page](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) states [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source condition labels, cycle indices, and aligned study inputs retain that license with this attribution. Operating-segment IDs are study proxies and are not a claim of documented independent physical runs. The repository provides results and acquisition/provenance information; original multirate signal archives are obtained from UCI.

## Third-party plotting resources

SciencePlots uses the MIT license at commit `b9b16959570bd2fbc9ff5118bacc423c3bddd592`. Baker Laboratory's figure skill and QA tools use Apache-2.0 at commit `ee5aaa47060ed116d7ae2d2429761ae6ab07f9e0`. Their original notices and license texts are retained beside the vendored resources. Neither resource is claimed as original study software, and neither implies Nature endorsement.

Original runtime environments, Python libraries, operating-system fonts, wheels, and DLLs are not redistributed. Install the documented dependencies from their own distributors under their respective terms.
