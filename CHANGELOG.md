# [1.1.0](https://github.com/wesleywxie/runpod-worker-comfy/compare/1.0.0...1.1.0) (2025-02-12)


### Features

* add libglib2.0-0 dependency ([86d9a65](https://github.com/wesleywxie/runpod-worker-comfy/commit/86d9a65f20040b1d3b9a9ea11176758884f7b286))

# 1.0.0 (2025-01-26)


### Bug Fixes

* added missing start command ([9a7ffdb](https://github.com/wesleywxie/runpod-worker-comfy/commit/9a7ffdb078d2f75194c86ed0b8c2d027592e52c3))
* check_server default values for delay and check-interval ([4945a9d](https://github.com/wesleywxie/runpod-worker-comfy/commit/4945a9d65b55aae9117591c8d64f9882d200478e))
* convert environment variables to int ([#70](https://github.com/wesleywxie/runpod-worker-comfy/issues/70)) ([7ab3d2a](https://github.com/wesleywxie/runpod-worker-comfy/commit/7ab3d2a234325c2a502002ea7bdee7df3e0c8dfe))
* create directories which are required to run ComfyUI ([#58](https://github.com/wesleywxie/runpod-worker-comfy/issues/58)) ([6edf62b](https://github.com/wesleywxie/runpod-worker-comfy/commit/6edf62b0f4cd99dba5c22dd76f51c886f57a28ed))
* don’t persist credentials ([1546420](https://github.com/wesleywxie/runpod-worker-comfy/commit/15464201b24de0746fe365e7635540330887a393))
* don't run ntpdate as this is not working in GitHub Actions ([2f7bd3f](https://github.com/wesleywxie/runpod-worker-comfy/commit/2f7bd3f71f24dd3b6ecc56f3a4c27bbc2d140eca))
* fixed the syntax error ([ac3dddb](https://github.com/wesleywxie/runpod-worker-comfy/commit/ac3dddb0939bb76a7118bf112e39b406afcebf41))
* fixed typo for ipadatper ([5d7c166](https://github.com/wesleywxie/runpod-worker-comfy/commit/5d7c1663ea559c5a17242f1b4f0e7c0b15986a16))
* got rid of syntax error ([c04de4d](https://github.com/wesleywxie/runpod-worker-comfy/commit/c04de4dea93dbe586a9a887e04907b33597ff73e))
* images in subfolders are not working, fixes [#12](https://github.com/wesleywxie/runpod-worker-comfy/issues/12) ([37480c2](https://github.com/wesleywxie/runpod-worker-comfy/commit/37480c2d217698f799f6388ff311b9f8c6c38804))
* path should be "loras" and not "lora" ([8e579f6](https://github.com/wesleywxie/runpod-worker-comfy/commit/8e579f63e18851b0be67bff7a42a8e8a46223f2b))
* put snapshot file in the right place ([69be408](https://github.com/wesleywxie/runpod-worker-comfy/commit/69be4087e80a0375b52b4ecbc642393827b58053))
* removed xl_more_art-full_v1 because civitai requires login now ([2e8e638](https://github.com/wesleywxie/runpod-worker-comfy/commit/2e8e63801a7672e4923eaad0c18a4b3e2c14d79c))
* return the output of "process_output_image" and access jobId correctly ([#11](https://github.com/wesleywxie/runpod-worker-comfy/issues/11)) ([dc655ea](https://github.com/wesleywxie/runpod-worker-comfy/commit/dc655ea0dd0b294703f52f6017ce095c3b411527))
* **semantic-release:** added .releaserc ([#21](https://github.com/wesleywxie/runpod-worker-comfy/issues/21)) ([12b763d](https://github.com/wesleywxie/runpod-worker-comfy/commit/12b763d8703ce07331a16d4013975f9edc4be3ff))
* start the container in all cases ([413707b](https://github.com/wesleywxie/runpod-worker-comfy/commit/413707bf130eb736afd682adac8b37fa64a5c9a4))
* update the version inside of semanticrelease ([d93e991](https://github.com/wesleywxie/runpod-worker-comfy/commit/d93e991b82251d62500e20c367a087d22d58b20a))
* updated path to "comfyui" ([37f66d0](https://github.com/wesleywxie/runpod-worker-comfy/commit/37f66d04b8c98810714ffbc761412f3fcdb1d861))
* use custom GITHUB_TOKEN to bypass branch protection ([9b6468a](https://github.com/wesleywxie/runpod-worker-comfy/commit/9b6468a40b8a476d7812423ff6fe7b73f5f91f1d))


### Features

* added default ComfyUI workflow ([fa6c385](https://github.com/wesleywxie/runpod-worker-comfy/commit/fa6c385e0dc9487655b42772bb6f3a5f5218864e))
* added FLUX.1 schnell & dev ([9170191](https://github.com/wesleywxie/runpod-worker-comfy/commit/9170191eccb65de2f17009f68952a18fc008fa6a))
* added runpod as local dependency ([9deae9f](https://github.com/wesleywxie/runpod-worker-comfy/commit/9deae9f5ec723b93540e6e2deac04b8650cf872a))
* added sensible defaults and default platform ([3f5162a](https://github.com/wesleywxie/runpod-worker-comfy/commit/3f5162af85ee7d0002ad65a7e324c3850e00a229))
* added unit tests for everthing, refactored the code to make it better testable, added test images ([a7492ec](https://github.com/wesleywxie/runpod-worker-comfy/commit/a7492ec8f289fc64b8e54c319f47804c0a15ae54))
* added xl_more_art-full_v1, improved comments ([9aea8ab](https://github.com/wesleywxie/runpod-worker-comfy/commit/9aea8abe1375f3d48aa9742c444b5242111e3121))
* automatically update latest version ([7d846e8](https://github.com/wesleywxie/runpod-worker-comfy/commit/7d846e8ca3edcea869db3e680f0b423b8a98cc4c))
* base64 image output ([#8](https://github.com/wesleywxie/runpod-worker-comfy/issues/8)) ([76bf0b1](https://github.com/wesleywxie/runpod-worker-comfy/commit/76bf0b166b992a208c53f5cb98bd20a7e3c7f933))
* build ipadater only ([d839c40](https://github.com/wesleywxie/runpod-worker-comfy/commit/d839c403e62b9980b0c7a27b6bfcf0c1df137bc7))
* example on how to configure the .env ([4ed5296](https://github.com/wesleywxie/runpod-worker-comfy/commit/4ed529601394e8a105d171ab1274737392da7df5))
* image-input, renamed "prompt" to "workflow", added "REFRESH_WORKER" ([#14](https://github.com/wesleywxie/runpod-worker-comfy/issues/14)) ([5f5e390](https://github.com/wesleywxie/runpod-worker-comfy/commit/5f5e390dfda9d3ef8ce9b5578aade1bee600bf5c))
* logs should be written to stdout so that we can see them inside the worker ([fc731ff](https://github.com/wesleywxie/runpod-worker-comfy/commit/fc731fffcd79af67cf6fcdf6a6d3df6b8e30c7b5))
* network-volume; execution time config; skip default images; access ComfyUI via web ([#35](https://github.com/wesleywxie/runpod-worker-comfy/issues/35)) ([070cde5](https://github.com/wesleywxie/runpod-worker-comfy/commit/070cde5460203e24e3fbf68c4ff6c9a9b7910f3f)), closes [#16](https://github.com/wesleywxie/runpod-worker-comfy/issues/16)
* remove sdxl and flux1-schnell. add ipadapter ([89122f5](https://github.com/wesleywxie/runpod-worker-comfy/commit/89122f542c6a795fd95572ba06803ec3bc1a30e9))
* run the worker locally ([#19](https://github.com/wesleywxie/runpod-worker-comfy/issues/19)) ([34eb32b](https://github.com/wesleywxie/runpod-worker-comfy/commit/34eb32b72455e6e628849e50405ed172d846d2d9))
* simplified and added compatibility with Windows ([9f41231](https://github.com/wesleywxie/runpod-worker-comfy/commit/9f412316a743f0539981b408c1ccd0692cff5c82))
* simplified input ([35c2341](https://github.com/wesleywxie/runpod-worker-comfy/commit/35c2341deca346d4e6df82c36e101b7495f3fc03))
* simplified input to just have "prompt", removed unused code ([0c3ccda](https://github.com/wesleywxie/runpod-worker-comfy/commit/0c3ccda9c5c8cdc56eae829bb358ceb532b36371))
* support ipadapter ([cd8b9da](https://github.com/wesleywxie/runpod-worker-comfy/commit/cd8b9dadb4695d74892c740819c770255de25f61))
* support sd3 ([#46](https://github.com/wesleywxie/runpod-worker-comfy/issues/46)) ([dde69d6](https://github.com/wesleywxie/runpod-worker-comfy/commit/dde69d6ca75eb7e4c5f01fd17e6da5b62f8a401f))
* updated path to "comfyui", added "ntpdate" to have the time of the container in sync with AWS ([2fda578](https://github.com/wesleywxie/runpod-worker-comfy/commit/2fda578d62460275abec11d6b2fbe5123d621d5f))
* use local ".env" to load env variables, mount "comfyui/output" to localhost so that people can see the generated images ([aa645a2](https://github.com/wesleywxie/runpod-worker-comfy/commit/aa645a233cd6951d296d68f7ddcf41b14b3f4cf9))
* use models from huggingface, not from local folder ([b1af369](https://github.com/wesleywxie/runpod-worker-comfy/commit/b1af369bb577c0aaba8875d8b2076e1888356929))
* wait until server is ready, wait until image generation is done, upload to s3 ([ecfec13](https://github.com/wesleywxie/runpod-worker-comfy/commit/ecfec1349da0d04ea5f21c82d8903e1a5bd3c923))


### BREAKING CHANGES

* we have 3 different images now instead of just one: base, sdxl and sd3

* ci: use branch name for creating dev releases

* ci: replace "/" with "-" to have a valid tag name

* ci: correctly handle the tag  name

* ci: build an image that contains sd3 using docker bake

* ci: use "set" instead of "args"

* ci: use "env" instead of "set"

* ci: use variables instead of args

* ci: set variables directly for the targets

* ci: write the secrets into the GITHUB_ENV

* ci: handle env variables correctly

* ci: use env variables from GitHub Variables

* ci: added back to env

* ci: print out env

* ci: adding the vars directly into the workflow

* ci: example workflow for sd3

* ci: renamed DOCKERHUB_REPO to DOCKERHUB_REPOSITORY

* ci: removed quotes for DOCKERHUB_REPOSITORY

* ci: only use DOCKERHUB_REPO in bake

* ci: added vars into sd3 target

* ci: added direct target

* ci: back to basics

* ci: multi-stage build to not expose the HUGGINGFACE_ACCESS_TOKEN

* ci: write everything into GITHUB_ENV again

* ci: use correct name for final stage

* ci: use correct runner

* fix: make sure to use the latest versions of all packages

* ci: simplified variables for all targets

* docs: added 3 images, updated build your own image

* docs: updated TOC

* ci: updated name

* ci: use docker bake to publish 3 images instead of just 1

# 1.0.0 (2025-01-24)


### Bug Fixes

* added missing start command ([9a7ffdb](https://github.com/wesleywxie/runpod-worker-comfy/commit/9a7ffdb078d2f75194c86ed0b8c2d027592e52c3))
* check_server default values for delay and check-interval ([4945a9d](https://github.com/wesleywxie/runpod-worker-comfy/commit/4945a9d65b55aae9117591c8d64f9882d200478e))
* convert environment variables to int ([#70](https://github.com/wesleywxie/runpod-worker-comfy/issues/70)) ([7ab3d2a](https://github.com/wesleywxie/runpod-worker-comfy/commit/7ab3d2a234325c2a502002ea7bdee7df3e0c8dfe))
* create directories which are required to run ComfyUI ([#58](https://github.com/wesleywxie/runpod-worker-comfy/issues/58)) ([6edf62b](https://github.com/wesleywxie/runpod-worker-comfy/commit/6edf62b0f4cd99dba5c22dd76f51c886f57a28ed))
* don’t persist credentials ([1546420](https://github.com/wesleywxie/runpod-worker-comfy/commit/15464201b24de0746fe365e7635540330887a393))
* don't run ntpdate as this is not working in GitHub Actions ([2f7bd3f](https://github.com/wesleywxie/runpod-worker-comfy/commit/2f7bd3f71f24dd3b6ecc56f3a4c27bbc2d140eca))
* fixed the syntax error ([ac3dddb](https://github.com/wesleywxie/runpod-worker-comfy/commit/ac3dddb0939bb76a7118bf112e39b406afcebf41))
* fixed typo for ipadatper ([5d7c166](https://github.com/wesleywxie/runpod-worker-comfy/commit/5d7c1663ea559c5a17242f1b4f0e7c0b15986a16))
* got rid of syntax error ([c04de4d](https://github.com/wesleywxie/runpod-worker-comfy/commit/c04de4dea93dbe586a9a887e04907b33597ff73e))
* images in subfolders are not working, fixes [#12](https://github.com/wesleywxie/runpod-worker-comfy/issues/12) ([37480c2](https://github.com/wesleywxie/runpod-worker-comfy/commit/37480c2d217698f799f6388ff311b9f8c6c38804))
* path should be "loras" and not "lora" ([8e579f6](https://github.com/wesleywxie/runpod-worker-comfy/commit/8e579f63e18851b0be67bff7a42a8e8a46223f2b))
* put snapshot file in the right place ([69be408](https://github.com/wesleywxie/runpod-worker-comfy/commit/69be4087e80a0375b52b4ecbc642393827b58053))
* removed xl_more_art-full_v1 because civitai requires login now ([2e8e638](https://github.com/wesleywxie/runpod-worker-comfy/commit/2e8e63801a7672e4923eaad0c18a4b3e2c14d79c))
* return the output of "process_output_image" and access jobId correctly ([#11](https://github.com/wesleywxie/runpod-worker-comfy/issues/11)) ([dc655ea](https://github.com/wesleywxie/runpod-worker-comfy/commit/dc655ea0dd0b294703f52f6017ce095c3b411527))
* **semantic-release:** added .releaserc ([#21](https://github.com/wesleywxie/runpod-worker-comfy/issues/21)) ([12b763d](https://github.com/wesleywxie/runpod-worker-comfy/commit/12b763d8703ce07331a16d4013975f9edc4be3ff))
* start the container in all cases ([413707b](https://github.com/wesleywxie/runpod-worker-comfy/commit/413707bf130eb736afd682adac8b37fa64a5c9a4))
* update the version inside of semanticrelease ([d93e991](https://github.com/wesleywxie/runpod-worker-comfy/commit/d93e991b82251d62500e20c367a087d22d58b20a))
* updated path to "comfyui" ([37f66d0](https://github.com/wesleywxie/runpod-worker-comfy/commit/37f66d04b8c98810714ffbc761412f3fcdb1d861))
* use custom GITHUB_TOKEN to bypass branch protection ([9b6468a](https://github.com/wesleywxie/runpod-worker-comfy/commit/9b6468a40b8a476d7812423ff6fe7b73f5f91f1d))


### Features

* added default ComfyUI workflow ([fa6c385](https://github.com/wesleywxie/runpod-worker-comfy/commit/fa6c385e0dc9487655b42772bb6f3a5f5218864e))
* added FLUX.1 schnell & dev ([9170191](https://github.com/wesleywxie/runpod-worker-comfy/commit/9170191eccb65de2f17009f68952a18fc008fa6a))
* added runpod as local dependency ([9deae9f](https://github.com/wesleywxie/runpod-worker-comfy/commit/9deae9f5ec723b93540e6e2deac04b8650cf872a))
* added sensible defaults and default platform ([3f5162a](https://github.com/wesleywxie/runpod-worker-comfy/commit/3f5162af85ee7d0002ad65a7e324c3850e00a229))
* added unit tests for everthing, refactored the code to make it better testable, added test images ([a7492ec](https://github.com/wesleywxie/runpod-worker-comfy/commit/a7492ec8f289fc64b8e54c319f47804c0a15ae54))
* added xl_more_art-full_v1, improved comments ([9aea8ab](https://github.com/wesleywxie/runpod-worker-comfy/commit/9aea8abe1375f3d48aa9742c444b5242111e3121))
* automatically update latest version ([7d846e8](https://github.com/wesleywxie/runpod-worker-comfy/commit/7d846e8ca3edcea869db3e680f0b423b8a98cc4c))
* base64 image output ([#8](https://github.com/wesleywxie/runpod-worker-comfy/issues/8)) ([76bf0b1](https://github.com/wesleywxie/runpod-worker-comfy/commit/76bf0b166b992a208c53f5cb98bd20a7e3c7f933))
* example on how to configure the .env ([4ed5296](https://github.com/wesleywxie/runpod-worker-comfy/commit/4ed529601394e8a105d171ab1274737392da7df5))
* image-input, renamed "prompt" to "workflow", added "REFRESH_WORKER" ([#14](https://github.com/wesleywxie/runpod-worker-comfy/issues/14)) ([5f5e390](https://github.com/wesleywxie/runpod-worker-comfy/commit/5f5e390dfda9d3ef8ce9b5578aade1bee600bf5c))
* logs should be written to stdout so that we can see them inside the worker ([fc731ff](https://github.com/wesleywxie/runpod-worker-comfy/commit/fc731fffcd79af67cf6fcdf6a6d3df6b8e30c7b5))
* network-volume; execution time config; skip default images; access ComfyUI via web ([#35](https://github.com/wesleywxie/runpod-worker-comfy/issues/35)) ([070cde5](https://github.com/wesleywxie/runpod-worker-comfy/commit/070cde5460203e24e3fbf68c4ff6c9a9b7910f3f)), closes [#16](https://github.com/wesleywxie/runpod-worker-comfy/issues/16)
* remove sdxl and flux1-schnell. add ipadapter ([89122f5](https://github.com/wesleywxie/runpod-worker-comfy/commit/89122f542c6a795fd95572ba06803ec3bc1a30e9))
* run the worker locally ([#19](https://github.com/wesleywxie/runpod-worker-comfy/issues/19)) ([34eb32b](https://github.com/wesleywxie/runpod-worker-comfy/commit/34eb32b72455e6e628849e50405ed172d846d2d9))
* simplified and added compatibility with Windows ([9f41231](https://github.com/wesleywxie/runpod-worker-comfy/commit/9f412316a743f0539981b408c1ccd0692cff5c82))
* simplified input ([35c2341](https://github.com/wesleywxie/runpod-worker-comfy/commit/35c2341deca346d4e6df82c36e101b7495f3fc03))
* simplified input to just have "prompt", removed unused code ([0c3ccda](https://github.com/wesleywxie/runpod-worker-comfy/commit/0c3ccda9c5c8cdc56eae829bb358ceb532b36371))
* support ipadapter ([cd8b9da](https://github.com/wesleywxie/runpod-worker-comfy/commit/cd8b9dadb4695d74892c740819c770255de25f61))
* support sd3 ([#46](https://github.com/wesleywxie/runpod-worker-comfy/issues/46)) ([dde69d6](https://github.com/wesleywxie/runpod-worker-comfy/commit/dde69d6ca75eb7e4c5f01fd17e6da5b62f8a401f))
* updated path to "comfyui", added "ntpdate" to have the time of the container in sync with AWS ([2fda578](https://github.com/wesleywxie/runpod-worker-comfy/commit/2fda578d62460275abec11d6b2fbe5123d621d5f))
* use local ".env" to load env variables, mount "comfyui/output" to localhost so that people can see the generated images ([aa645a2](https://github.com/wesleywxie/runpod-worker-comfy/commit/aa645a233cd6951d296d68f7ddcf41b14b3f4cf9))
* use models from huggingface, not from local folder ([b1af369](https://github.com/wesleywxie/runpod-worker-comfy/commit/b1af369bb577c0aaba8875d8b2076e1888356929))
* wait until server is ready, wait until image generation is done, upload to s3 ([ecfec13](https://github.com/wesleywxie/runpod-worker-comfy/commit/ecfec1349da0d04ea5f21c82d8903e1a5bd3c923))


### BREAKING CHANGES

* we have 3 different images now instead of just one: base, sdxl and sd3

* ci: use branch name for creating dev releases

* ci: replace "/" with "-" to have a valid tag name

* ci: correctly handle the tag  name

* ci: build an image that contains sd3 using docker bake

* ci: use "set" instead of "args"

* ci: use "env" instead of "set"

* ci: use variables instead of args

* ci: set variables directly for the targets

* ci: write the secrets into the GITHUB_ENV

* ci: handle env variables correctly

* ci: use env variables from GitHub Variables

* ci: added back to env

* ci: print out env

* ci: adding the vars directly into the workflow

* ci: example workflow for sd3

* ci: renamed DOCKERHUB_REPO to DOCKERHUB_REPOSITORY

* ci: removed quotes for DOCKERHUB_REPOSITORY

* ci: only use DOCKERHUB_REPO in bake

* ci: added vars into sd3 target

* ci: added direct target

* ci: back to basics

* ci: multi-stage build to not expose the HUGGINGFACE_ACCESS_TOKEN

* ci: write everything into GITHUB_ENV again

* ci: use correct name for final stage

* ci: use correct runner

* fix: make sure to use the latest versions of all packages

* ci: simplified variables for all targets

* docs: added 3 images, updated build your own image

* docs: updated TOC

* ci: updated name

* ci: use docker bake to publish 3 images instead of just 1

# 1.0.0 (2025-01-24)


### Bug Fixes

* added missing start command ([9a7ffdb](https://github.com/wesleywxie/runpod-worker-comfy/commit/9a7ffdb078d2f75194c86ed0b8c2d027592e52c3))
* check_server default values for delay and check-interval ([4945a9d](https://github.com/wesleywxie/runpod-worker-comfy/commit/4945a9d65b55aae9117591c8d64f9882d200478e))
* convert environment variables to int ([#70](https://github.com/wesleywxie/runpod-worker-comfy/issues/70)) ([7ab3d2a](https://github.com/wesleywxie/runpod-worker-comfy/commit/7ab3d2a234325c2a502002ea7bdee7df3e0c8dfe))
* create directories which are required to run ComfyUI ([#58](https://github.com/wesleywxie/runpod-worker-comfy/issues/58)) ([6edf62b](https://github.com/wesleywxie/runpod-worker-comfy/commit/6edf62b0f4cd99dba5c22dd76f51c886f57a28ed))
* don’t persist credentials ([1546420](https://github.com/wesleywxie/runpod-worker-comfy/commit/15464201b24de0746fe365e7635540330887a393))
* don't run ntpdate as this is not working in GitHub Actions ([2f7bd3f](https://github.com/wesleywxie/runpod-worker-comfy/commit/2f7bd3f71f24dd3b6ecc56f3a4c27bbc2d140eca))
* fixed the syntax error ([ac3dddb](https://github.com/wesleywxie/runpod-worker-comfy/commit/ac3dddb0939bb76a7118bf112e39b406afcebf41))
* got rid of syntax error ([c04de4d](https://github.com/wesleywxie/runpod-worker-comfy/commit/c04de4dea93dbe586a9a887e04907b33597ff73e))
* images in subfolders are not working, fixes [#12](https://github.com/wesleywxie/runpod-worker-comfy/issues/12) ([37480c2](https://github.com/wesleywxie/runpod-worker-comfy/commit/37480c2d217698f799f6388ff311b9f8c6c38804))
* path should be "loras" and not "lora" ([8e579f6](https://github.com/wesleywxie/runpod-worker-comfy/commit/8e579f63e18851b0be67bff7a42a8e8a46223f2b))
* put snapshot file in the right place ([69be408](https://github.com/wesleywxie/runpod-worker-comfy/commit/69be4087e80a0375b52b4ecbc642393827b58053))
* removed xl_more_art-full_v1 because civitai requires login now ([2e8e638](https://github.com/wesleywxie/runpod-worker-comfy/commit/2e8e63801a7672e4923eaad0c18a4b3e2c14d79c))
* return the output of "process_output_image" and access jobId correctly ([#11](https://github.com/wesleywxie/runpod-worker-comfy/issues/11)) ([dc655ea](https://github.com/wesleywxie/runpod-worker-comfy/commit/dc655ea0dd0b294703f52f6017ce095c3b411527))
* **semantic-release:** added .releaserc ([#21](https://github.com/wesleywxie/runpod-worker-comfy/issues/21)) ([12b763d](https://github.com/wesleywxie/runpod-worker-comfy/commit/12b763d8703ce07331a16d4013975f9edc4be3ff))
* start the container in all cases ([413707b](https://github.com/wesleywxie/runpod-worker-comfy/commit/413707bf130eb736afd682adac8b37fa64a5c9a4))
* update the version inside of semanticrelease ([d93e991](https://github.com/wesleywxie/runpod-worker-comfy/commit/d93e991b82251d62500e20c367a087d22d58b20a))
* updated path to "comfyui" ([37f66d0](https://github.com/wesleywxie/runpod-worker-comfy/commit/37f66d04b8c98810714ffbc761412f3fcdb1d861))
* use custom GITHUB_TOKEN to bypass branch protection ([9b6468a](https://github.com/wesleywxie/runpod-worker-comfy/commit/9b6468a40b8a476d7812423ff6fe7b73f5f91f1d))


### Features

* added default ComfyUI workflow ([fa6c385](https://github.com/wesleywxie/runpod-worker-comfy/commit/fa6c385e0dc9487655b42772bb6f3a5f5218864e))
* added FLUX.1 schnell & dev ([9170191](https://github.com/wesleywxie/runpod-worker-comfy/commit/9170191eccb65de2f17009f68952a18fc008fa6a))
* added runpod as local dependency ([9deae9f](https://github.com/wesleywxie/runpod-worker-comfy/commit/9deae9f5ec723b93540e6e2deac04b8650cf872a))
* added sensible defaults and default platform ([3f5162a](https://github.com/wesleywxie/runpod-worker-comfy/commit/3f5162af85ee7d0002ad65a7e324c3850e00a229))
* added unit tests for everthing, refactored the code to make it better testable, added test images ([a7492ec](https://github.com/wesleywxie/runpod-worker-comfy/commit/a7492ec8f289fc64b8e54c319f47804c0a15ae54))
* added xl_more_art-full_v1, improved comments ([9aea8ab](https://github.com/wesleywxie/runpod-worker-comfy/commit/9aea8abe1375f3d48aa9742c444b5242111e3121))
* automatically update latest version ([7d846e8](https://github.com/wesleywxie/runpod-worker-comfy/commit/7d846e8ca3edcea869db3e680f0b423b8a98cc4c))
* base64 image output ([#8](https://github.com/wesleywxie/runpod-worker-comfy/issues/8)) ([76bf0b1](https://github.com/wesleywxie/runpod-worker-comfy/commit/76bf0b166b992a208c53f5cb98bd20a7e3c7f933))
* example on how to configure the .env ([4ed5296](https://github.com/wesleywxie/runpod-worker-comfy/commit/4ed529601394e8a105d171ab1274737392da7df5))
* image-input, renamed "prompt" to "workflow", added "REFRESH_WORKER" ([#14](https://github.com/wesleywxie/runpod-worker-comfy/issues/14)) ([5f5e390](https://github.com/wesleywxie/runpod-worker-comfy/commit/5f5e390dfda9d3ef8ce9b5578aade1bee600bf5c))
* logs should be written to stdout so that we can see them inside the worker ([fc731ff](https://github.com/wesleywxie/runpod-worker-comfy/commit/fc731fffcd79af67cf6fcdf6a6d3df6b8e30c7b5))
* network-volume; execution time config; skip default images; access ComfyUI via web ([#35](https://github.com/wesleywxie/runpod-worker-comfy/issues/35)) ([070cde5](https://github.com/wesleywxie/runpod-worker-comfy/commit/070cde5460203e24e3fbf68c4ff6c9a9b7910f3f)), closes [#16](https://github.com/wesleywxie/runpod-worker-comfy/issues/16)
* remove sdxl and flux1-schnell. add ipadapter ([89122f5](https://github.com/wesleywxie/runpod-worker-comfy/commit/89122f542c6a795fd95572ba06803ec3bc1a30e9))
* run the worker locally ([#19](https://github.com/wesleywxie/runpod-worker-comfy/issues/19)) ([34eb32b](https://github.com/wesleywxie/runpod-worker-comfy/commit/34eb32b72455e6e628849e50405ed172d846d2d9))
* simplified and added compatibility with Windows ([9f41231](https://github.com/wesleywxie/runpod-worker-comfy/commit/9f412316a743f0539981b408c1ccd0692cff5c82))
* simplified input ([35c2341](https://github.com/wesleywxie/runpod-worker-comfy/commit/35c2341deca346d4e6df82c36e101b7495f3fc03))
* simplified input to just have "prompt", removed unused code ([0c3ccda](https://github.com/wesleywxie/runpod-worker-comfy/commit/0c3ccda9c5c8cdc56eae829bb358ceb532b36371))
* support ipadapter ([cd8b9da](https://github.com/wesleywxie/runpod-worker-comfy/commit/cd8b9dadb4695d74892c740819c770255de25f61))
* support sd3 ([#46](https://github.com/wesleywxie/runpod-worker-comfy/issues/46)) ([dde69d6](https://github.com/wesleywxie/runpod-worker-comfy/commit/dde69d6ca75eb7e4c5f01fd17e6da5b62f8a401f))
* updated path to "comfyui", added "ntpdate" to have the time of the container in sync with AWS ([2fda578](https://github.com/wesleywxie/runpod-worker-comfy/commit/2fda578d62460275abec11d6b2fbe5123d621d5f))
* use local ".env" to load env variables, mount "comfyui/output" to localhost so that people can see the generated images ([aa645a2](https://github.com/wesleywxie/runpod-worker-comfy/commit/aa645a233cd6951d296d68f7ddcf41b14b3f4cf9))
* use models from huggingface, not from local folder ([b1af369](https://github.com/wesleywxie/runpod-worker-comfy/commit/b1af369bb577c0aaba8875d8b2076e1888356929))
* wait until server is ready, wait until image generation is done, upload to s3 ([ecfec13](https://github.com/wesleywxie/runpod-worker-comfy/commit/ecfec1349da0d04ea5f21c82d8903e1a5bd3c923))


### BREAKING CHANGES

* we have 3 different images now instead of just one: base, sdxl and sd3

* ci: use branch name for creating dev releases

* ci: replace "/" with "-" to have a valid tag name

* ci: correctly handle the tag  name

* ci: build an image that contains sd3 using docker bake

* ci: use "set" instead of "args"

* ci: use "env" instead of "set"

* ci: use variables instead of args

* ci: set variables directly for the targets

* ci: write the secrets into the GITHUB_ENV

* ci: handle env variables correctly

* ci: use env variables from GitHub Variables

* ci: added back to env

* ci: print out env

* ci: adding the vars directly into the workflow

* ci: example workflow for sd3

* ci: renamed DOCKERHUB_REPO to DOCKERHUB_REPOSITORY

* ci: removed quotes for DOCKERHUB_REPOSITORY

* ci: only use DOCKERHUB_REPO in bake

* ci: added vars into sd3 target

* ci: added direct target

* ci: back to basics

* ci: multi-stage build to not expose the HUGGINGFACE_ACCESS_TOKEN

* ci: write everything into GITHUB_ENV again

* ci: use correct name for final stage

* ci: use correct runner

* fix: make sure to use the latest versions of all packages

* ci: simplified variables for all targets

* docs: added 3 images, updated build your own image

* docs: updated TOC

* ci: updated name

* ci: use docker bake to publish 3 images instead of just 1

# 1.0.0 (2025-01-24)


### Bug Fixes

* added missing start command ([9a7ffdb](https://github.com/wesleywxie/runpod-worker-comfy/commit/9a7ffdb078d2f75194c86ed0b8c2d027592e52c3))
* check_server default values for delay and check-interval ([4945a9d](https://github.com/wesleywxie/runpod-worker-comfy/commit/4945a9d65b55aae9117591c8d64f9882d200478e))
* convert environment variables to int ([#70](https://github.com/wesleywxie/runpod-worker-comfy/issues/70)) ([7ab3d2a](https://github.com/wesleywxie/runpod-worker-comfy/commit/7ab3d2a234325c2a502002ea7bdee7df3e0c8dfe))
* create directories which are required to run ComfyUI ([#58](https://github.com/wesleywxie/runpod-worker-comfy/issues/58)) ([6edf62b](https://github.com/wesleywxie/runpod-worker-comfy/commit/6edf62b0f4cd99dba5c22dd76f51c886f57a28ed))
* don’t persist credentials ([1546420](https://github.com/wesleywxie/runpod-worker-comfy/commit/15464201b24de0746fe365e7635540330887a393))
* don't run ntpdate as this is not working in GitHub Actions ([2f7bd3f](https://github.com/wesleywxie/runpod-worker-comfy/commit/2f7bd3f71f24dd3b6ecc56f3a4c27bbc2d140eca))
* got rid of syntax error ([c04de4d](https://github.com/wesleywxie/runpod-worker-comfy/commit/c04de4dea93dbe586a9a887e04907b33597ff73e))
* images in subfolders are not working, fixes [#12](https://github.com/wesleywxie/runpod-worker-comfy/issues/12) ([37480c2](https://github.com/wesleywxie/runpod-worker-comfy/commit/37480c2d217698f799f6388ff311b9f8c6c38804))
* path should be "loras" and not "lora" ([8e579f6](https://github.com/wesleywxie/runpod-worker-comfy/commit/8e579f63e18851b0be67bff7a42a8e8a46223f2b))
* put snapshot file in the right place ([69be408](https://github.com/wesleywxie/runpod-worker-comfy/commit/69be4087e80a0375b52b4ecbc642393827b58053))
* removed xl_more_art-full_v1 because civitai requires login now ([2e8e638](https://github.com/wesleywxie/runpod-worker-comfy/commit/2e8e63801a7672e4923eaad0c18a4b3e2c14d79c))
* return the output of "process_output_image" and access jobId correctly ([#11](https://github.com/wesleywxie/runpod-worker-comfy/issues/11)) ([dc655ea](https://github.com/wesleywxie/runpod-worker-comfy/commit/dc655ea0dd0b294703f52f6017ce095c3b411527))
* **semantic-release:** added .releaserc ([#21](https://github.com/wesleywxie/runpod-worker-comfy/issues/21)) ([12b763d](https://github.com/wesleywxie/runpod-worker-comfy/commit/12b763d8703ce07331a16d4013975f9edc4be3ff))
* start the container in all cases ([413707b](https://github.com/wesleywxie/runpod-worker-comfy/commit/413707bf130eb736afd682adac8b37fa64a5c9a4))
* update the version inside of semanticrelease ([d93e991](https://github.com/wesleywxie/runpod-worker-comfy/commit/d93e991b82251d62500e20c367a087d22d58b20a))
* updated path to "comfyui" ([37f66d0](https://github.com/wesleywxie/runpod-worker-comfy/commit/37f66d04b8c98810714ffbc761412f3fcdb1d861))
* use custom GITHUB_TOKEN to bypass branch protection ([9b6468a](https://github.com/wesleywxie/runpod-worker-comfy/commit/9b6468a40b8a476d7812423ff6fe7b73f5f91f1d))


### Features

* added default ComfyUI workflow ([fa6c385](https://github.com/wesleywxie/runpod-worker-comfy/commit/fa6c385e0dc9487655b42772bb6f3a5f5218864e))
* added FLUX.1 schnell & dev ([9170191](https://github.com/wesleywxie/runpod-worker-comfy/commit/9170191eccb65de2f17009f68952a18fc008fa6a))
* added runpod as local dependency ([9deae9f](https://github.com/wesleywxie/runpod-worker-comfy/commit/9deae9f5ec723b93540e6e2deac04b8650cf872a))
* added sensible defaults and default platform ([3f5162a](https://github.com/wesleywxie/runpod-worker-comfy/commit/3f5162af85ee7d0002ad65a7e324c3850e00a229))
* added unit tests for everthing, refactored the code to make it better testable, added test images ([a7492ec](https://github.com/wesleywxie/runpod-worker-comfy/commit/a7492ec8f289fc64b8e54c319f47804c0a15ae54))
* added xl_more_art-full_v1, improved comments ([9aea8ab](https://github.com/wesleywxie/runpod-worker-comfy/commit/9aea8abe1375f3d48aa9742c444b5242111e3121))
* automatically update latest version ([7d846e8](https://github.com/wesleywxie/runpod-worker-comfy/commit/7d846e8ca3edcea869db3e680f0b423b8a98cc4c))
* base64 image output ([#8](https://github.com/wesleywxie/runpod-worker-comfy/issues/8)) ([76bf0b1](https://github.com/wesleywxie/runpod-worker-comfy/commit/76bf0b166b992a208c53f5cb98bd20a7e3c7f933))
* example on how to configure the .env ([4ed5296](https://github.com/wesleywxie/runpod-worker-comfy/commit/4ed529601394e8a105d171ab1274737392da7df5))
* image-input, renamed "prompt" to "workflow", added "REFRESH_WORKER" ([#14](https://github.com/wesleywxie/runpod-worker-comfy/issues/14)) ([5f5e390](https://github.com/wesleywxie/runpod-worker-comfy/commit/5f5e390dfda9d3ef8ce9b5578aade1bee600bf5c))
* logs should be written to stdout so that we can see them inside the worker ([fc731ff](https://github.com/wesleywxie/runpod-worker-comfy/commit/fc731fffcd79af67cf6fcdf6a6d3df6b8e30c7b5))
* network-volume; execution time config; skip default images; access ComfyUI via web ([#35](https://github.com/wesleywxie/runpod-worker-comfy/issues/35)) ([070cde5](https://github.com/wesleywxie/runpod-worker-comfy/commit/070cde5460203e24e3fbf68c4ff6c9a9b7910f3f)), closes [#16](https://github.com/wesleywxie/runpod-worker-comfy/issues/16)
* remove sdxl and flux1-schnell. add ipadapter ([89122f5](https://github.com/wesleywxie/runpod-worker-comfy/commit/89122f542c6a795fd95572ba06803ec3bc1a30e9))
* run the worker locally ([#19](https://github.com/wesleywxie/runpod-worker-comfy/issues/19)) ([34eb32b](https://github.com/wesleywxie/runpod-worker-comfy/commit/34eb32b72455e6e628849e50405ed172d846d2d9))
* simplified and added compatibility with Windows ([9f41231](https://github.com/wesleywxie/runpod-worker-comfy/commit/9f412316a743f0539981b408c1ccd0692cff5c82))
* simplified input ([35c2341](https://github.com/wesleywxie/runpod-worker-comfy/commit/35c2341deca346d4e6df82c36e101b7495f3fc03))
* simplified input to just have "prompt", removed unused code ([0c3ccda](https://github.com/wesleywxie/runpod-worker-comfy/commit/0c3ccda9c5c8cdc56eae829bb358ceb532b36371))
* support ipadapter ([cd8b9da](https://github.com/wesleywxie/runpod-worker-comfy/commit/cd8b9dadb4695d74892c740819c770255de25f61))
* support sd3 ([#46](https://github.com/wesleywxie/runpod-worker-comfy/issues/46)) ([dde69d6](https://github.com/wesleywxie/runpod-worker-comfy/commit/dde69d6ca75eb7e4c5f01fd17e6da5b62f8a401f))
* updated path to "comfyui", added "ntpdate" to have the time of the container in sync with AWS ([2fda578](https://github.com/wesleywxie/runpod-worker-comfy/commit/2fda578d62460275abec11d6b2fbe5123d621d5f))
* use local ".env" to load env variables, mount "comfyui/output" to localhost so that people can see the generated images ([aa645a2](https://github.com/wesleywxie/runpod-worker-comfy/commit/aa645a233cd6951d296d68f7ddcf41b14b3f4cf9))
* use models from huggingface, not from local folder ([b1af369](https://github.com/wesleywxie/runpod-worker-comfy/commit/b1af369bb577c0aaba8875d8b2076e1888356929))
* wait until server is ready, wait until image generation is done, upload to s3 ([ecfec13](https://github.com/wesleywxie/runpod-worker-comfy/commit/ecfec1349da0d04ea5f21c82d8903e1a5bd3c923))


### BREAKING CHANGES

* we have 3 different images now instead of just one: base, sdxl and sd3

* ci: use branch name for creating dev releases

* ci: replace "/" with "-" to have a valid tag name

* ci: correctly handle the tag  name

* ci: build an image that contains sd3 using docker bake

* ci: use "set" instead of "args"

* ci: use "env" instead of "set"

* ci: use variables instead of args

* ci: set variables directly for the targets

* ci: write the secrets into the GITHUB_ENV

* ci: handle env variables correctly

* ci: use env variables from GitHub Variables

* ci: added back to env

* ci: print out env

* ci: adding the vars directly into the workflow

* ci: example workflow for sd3

* ci: renamed DOCKERHUB_REPO to DOCKERHUB_REPOSITORY

* ci: removed quotes for DOCKERHUB_REPOSITORY

* ci: only use DOCKERHUB_REPO in bake

* ci: added vars into sd3 target

* ci: added direct target

* ci: back to basics

* ci: multi-stage build to not expose the HUGGINGFACE_ACCESS_TOKEN

* ci: write everything into GITHUB_ENV again

* ci: use correct name for final stage

* ci: use correct runner

* fix: make sure to use the latest versions of all packages

* ci: simplified variables for all targets

* docs: added 3 images, updated build your own image

* docs: updated TOC

* ci: updated name

* ci: use docker bake to publish 3 images instead of just 1

# 1.0.0 (2025-01-24)


### Bug Fixes

* added missing start command ([9a7ffdb](https://github.com/wesleywxie/runpod-worker-comfy/commit/9a7ffdb078d2f75194c86ed0b8c2d027592e52c3))
* check_server default values for delay and check-interval ([4945a9d](https://github.com/wesleywxie/runpod-worker-comfy/commit/4945a9d65b55aae9117591c8d64f9882d200478e))
* convert environment variables to int ([#70](https://github.com/wesleywxie/runpod-worker-comfy/issues/70)) ([7ab3d2a](https://github.com/wesleywxie/runpod-worker-comfy/commit/7ab3d2a234325c2a502002ea7bdee7df3e0c8dfe))
* create directories which are required to run ComfyUI ([#58](https://github.com/wesleywxie/runpod-worker-comfy/issues/58)) ([6edf62b](https://github.com/wesleywxie/runpod-worker-comfy/commit/6edf62b0f4cd99dba5c22dd76f51c886f57a28ed))
* don’t persist credentials ([1546420](https://github.com/wesleywxie/runpod-worker-comfy/commit/15464201b24de0746fe365e7635540330887a393))
* don't run ntpdate as this is not working in GitHub Actions ([2f7bd3f](https://github.com/wesleywxie/runpod-worker-comfy/commit/2f7bd3f71f24dd3b6ecc56f3a4c27bbc2d140eca))
* got rid of syntax error ([c04de4d](https://github.com/wesleywxie/runpod-worker-comfy/commit/c04de4dea93dbe586a9a887e04907b33597ff73e))
* images in subfolders are not working, fixes [#12](https://github.com/wesleywxie/runpod-worker-comfy/issues/12) ([37480c2](https://github.com/wesleywxie/runpod-worker-comfy/commit/37480c2d217698f799f6388ff311b9f8c6c38804))
* path should be "loras" and not "lora" ([8e579f6](https://github.com/wesleywxie/runpod-worker-comfy/commit/8e579f63e18851b0be67bff7a42a8e8a46223f2b))
* put snapshot file in the right place ([69be408](https://github.com/wesleywxie/runpod-worker-comfy/commit/69be4087e80a0375b52b4ecbc642393827b58053))
* removed xl_more_art-full_v1 because civitai requires login now ([2e8e638](https://github.com/wesleywxie/runpod-worker-comfy/commit/2e8e63801a7672e4923eaad0c18a4b3e2c14d79c))
* return the output of "process_output_image" and access jobId correctly ([#11](https://github.com/wesleywxie/runpod-worker-comfy/issues/11)) ([dc655ea](https://github.com/wesleywxie/runpod-worker-comfy/commit/dc655ea0dd0b294703f52f6017ce095c3b411527))
* **semantic-release:** added .releaserc ([#21](https://github.com/wesleywxie/runpod-worker-comfy/issues/21)) ([12b763d](https://github.com/wesleywxie/runpod-worker-comfy/commit/12b763d8703ce07331a16d4013975f9edc4be3ff))
* start the container in all cases ([413707b](https://github.com/wesleywxie/runpod-worker-comfy/commit/413707bf130eb736afd682adac8b37fa64a5c9a4))
* update the version inside of semanticrelease ([d93e991](https://github.com/wesleywxie/runpod-worker-comfy/commit/d93e991b82251d62500e20c367a087d22d58b20a))
* updated path to "comfyui" ([37f66d0](https://github.com/wesleywxie/runpod-worker-comfy/commit/37f66d04b8c98810714ffbc761412f3fcdb1d861))
* use custom GITHUB_TOKEN to bypass branch protection ([9b6468a](https://github.com/wesleywxie/runpod-worker-comfy/commit/9b6468a40b8a476d7812423ff6fe7b73f5f91f1d))


### Features

* added default ComfyUI workflow ([fa6c385](https://github.com/wesleywxie/runpod-worker-comfy/commit/fa6c385e0dc9487655b42772bb6f3a5f5218864e))
* added FLUX.1 schnell & dev ([9170191](https://github.com/wesleywxie/runpod-worker-comfy/commit/9170191eccb65de2f17009f68952a18fc008fa6a))
* added runpod as local dependency ([9deae9f](https://github.com/wesleywxie/runpod-worker-comfy/commit/9deae9f5ec723b93540e6e2deac04b8650cf872a))
* added sensible defaults and default platform ([3f5162a](https://github.com/wesleywxie/runpod-worker-comfy/commit/3f5162af85ee7d0002ad65a7e324c3850e00a229))
* added unit tests for everthing, refactored the code to make it better testable, added test images ([a7492ec](https://github.com/wesleywxie/runpod-worker-comfy/commit/a7492ec8f289fc64b8e54c319f47804c0a15ae54))
* added xl_more_art-full_v1, improved comments ([9aea8ab](https://github.com/wesleywxie/runpod-worker-comfy/commit/9aea8abe1375f3d48aa9742c444b5242111e3121))
* automatically update latest version ([7d846e8](https://github.com/wesleywxie/runpod-worker-comfy/commit/7d846e8ca3edcea869db3e680f0b423b8a98cc4c))
* base64 image output ([#8](https://github.com/wesleywxie/runpod-worker-comfy/issues/8)) ([76bf0b1](https://github.com/wesleywxie/runpod-worker-comfy/commit/76bf0b166b992a208c53f5cb98bd20a7e3c7f933))
* example on how to configure the .env ([4ed5296](https://github.com/wesleywxie/runpod-worker-comfy/commit/4ed529601394e8a105d171ab1274737392da7df5))
* image-input, renamed "prompt" to "workflow", added "REFRESH_WORKER" ([#14](https://github.com/wesleywxie/runpod-worker-comfy/issues/14)) ([5f5e390](https://github.com/wesleywxie/runpod-worker-comfy/commit/5f5e390dfda9d3ef8ce9b5578aade1bee600bf5c))
* logs should be written to stdout so that we can see them inside the worker ([fc731ff](https://github.com/wesleywxie/runpod-worker-comfy/commit/fc731fffcd79af67cf6fcdf6a6d3df6b8e30c7b5))
* network-volume; execution time config; skip default images; access ComfyUI via web ([#35](https://github.com/wesleywxie/runpod-worker-comfy/issues/35)) ([070cde5](https://github.com/wesleywxie/runpod-worker-comfy/commit/070cde5460203e24e3fbf68c4ff6c9a9b7910f3f)), closes [#16](https://github.com/wesleywxie/runpod-worker-comfy/issues/16)
* remove sdxl and flux1-schnell. add ipadapter ([89122f5](https://github.com/wesleywxie/runpod-worker-comfy/commit/89122f542c6a795fd95572ba06803ec3bc1a30e9))
* run the worker locally ([#19](https://github.com/wesleywxie/runpod-worker-comfy/issues/19)) ([34eb32b](https://github.com/wesleywxie/runpod-worker-comfy/commit/34eb32b72455e6e628849e50405ed172d846d2d9))
* simplified and added compatibility with Windows ([9f41231](https://github.com/wesleywxie/runpod-worker-comfy/commit/9f412316a743f0539981b408c1ccd0692cff5c82))
* simplified input ([35c2341](https://github.com/wesleywxie/runpod-worker-comfy/commit/35c2341deca346d4e6df82c36e101b7495f3fc03))
* simplified input to just have "prompt", removed unused code ([0c3ccda](https://github.com/wesleywxie/runpod-worker-comfy/commit/0c3ccda9c5c8cdc56eae829bb358ceb532b36371))
* support ipadapter ([cd8b9da](https://github.com/wesleywxie/runpod-worker-comfy/commit/cd8b9dadb4695d74892c740819c770255de25f61))
* support sd3 ([#46](https://github.com/wesleywxie/runpod-worker-comfy/issues/46)) ([dde69d6](https://github.com/wesleywxie/runpod-worker-comfy/commit/dde69d6ca75eb7e4c5f01fd17e6da5b62f8a401f))
* updated path to "comfyui", added "ntpdate" to have the time of the container in sync with AWS ([2fda578](https://github.com/wesleywxie/runpod-worker-comfy/commit/2fda578d62460275abec11d6b2fbe5123d621d5f))
* use local ".env" to load env variables, mount "comfyui/output" to localhost so that people can see the generated images ([aa645a2](https://github.com/wesleywxie/runpod-worker-comfy/commit/aa645a233cd6951d296d68f7ddcf41b14b3f4cf9))
* use models from huggingface, not from local folder ([b1af369](https://github.com/wesleywxie/runpod-worker-comfy/commit/b1af369bb577c0aaba8875d8b2076e1888356929))
* wait until server is ready, wait until image generation is done, upload to s3 ([ecfec13](https://github.com/wesleywxie/runpod-worker-comfy/commit/ecfec1349da0d04ea5f21c82d8903e1a5bd3c923))


### BREAKING CHANGES

* we have 3 different images now instead of just one: base, sdxl and sd3

* ci: use branch name for creating dev releases

* ci: replace "/" with "-" to have a valid tag name

* ci: correctly handle the tag  name

* ci: build an image that contains sd3 using docker bake

* ci: use "set" instead of "args"

* ci: use "env" instead of "set"

* ci: use variables instead of args

* ci: set variables directly for the targets

* ci: write the secrets into the GITHUB_ENV

* ci: handle env variables correctly

* ci: use env variables from GitHub Variables

* ci: added back to env

* ci: print out env

* ci: adding the vars directly into the workflow

* ci: example workflow for sd3

* ci: renamed DOCKERHUB_REPO to DOCKERHUB_REPOSITORY

* ci: removed quotes for DOCKERHUB_REPOSITORY

* ci: only use DOCKERHUB_REPO in bake

* ci: added vars into sd3 target

* ci: added direct target

* ci: back to basics

* ci: multi-stage build to not expose the HUGGINGFACE_ACCESS_TOKEN

* ci: write everything into GITHUB_ENV again

* ci: use correct name for final stage

* ci: use correct runner

* fix: make sure to use the latest versions of all packages

* ci: simplified variables for all targets

* docs: added 3 images, updated build your own image

* docs: updated TOC

* ci: updated name

* ci: use docker bake to publish 3 images instead of just 1

# [3.4.0](https://github.com/blib-la/runpod-worker-comfy/compare/3.3.0...3.4.0) (2024-11-19)


### Bug Fixes

* start the container in all cases ([413707b](https://github.com/blib-la/runpod-worker-comfy/commit/413707bf130eb736afd682adac8b37fa64a5c9a4))


### Features

* simplified and added compatibility with Windows ([9f41231](https://github.com/blib-la/runpod-worker-comfy/commit/9f412316a743f0539981b408c1ccd0692cff5c82))

# [3.3.0](https://github.com/blib-la/runpod-worker-comfy/compare/3.2.1...3.3.0) (2024-11-18)


### Bug Fixes

* added missing start command ([9a7ffdb](https://github.com/blib-la/runpod-worker-comfy/commit/9a7ffdb078d2f75194c86ed0b8c2d027592e52c3))


### Features

* added sensible defaults and default platform ([3f5162a](https://github.com/blib-la/runpod-worker-comfy/commit/3f5162af85ee7d0002ad65a7e324c3850e00a229))

## [3.2.1](https://github.com/blib-la/runpod-worker-comfy/compare/3.2.0...3.2.1) (2024-11-18)


### Bug Fixes

* update the version inside of semanticrelease ([d93e991](https://github.com/blib-la/runpod-worker-comfy/commit/d93e991b82251d62500e20c367a087d22d58b20a))

# [3.2.0](https://github.com/blib-la/runpod-worker-comfy/compare/3.1.2...3.2.0) (2024-11-18)


### Features

* automatically update latest version ([7d846e8](https://github.com/blib-la/runpod-worker-comfy/commit/7d846e8ca3edcea869db3e680f0b423b8a98cc4c))

## [3.1.2](https://github.com/blib-la/runpod-worker-comfy/compare/3.1.1...3.1.2) (2024-11-10)


### Bug Fixes

* convert environment variables to int ([#70](https://github.com/blib-la/runpod-worker-comfy/issues/70)) ([7ab3d2a](https://github.com/blib-la/runpod-worker-comfy/commit/7ab3d2a234325c2a502002ea7bdee7df3e0c8dfe))

## [3.1.1](https://github.com/blib-la/runpod-worker-comfy/compare/3.1.0...3.1.1) (2024-11-10)


### Bug Fixes

* create directories which are required to run ComfyUI ([#58](https://github.com/blib-la/runpod-worker-comfy/issues/58)) ([6edf62b](https://github.com/blib-la/runpod-worker-comfy/commit/6edf62b0f4cd99dba5c22dd76f51c886f57a28ed))

# [3.1.0](https://github.com/blib-la/runpod-worker-comfy/compare/3.0.0...3.1.0) (2024-08-19)


### Features

* added FLUX.1 schnell & dev ([9170191](https://github.com/blib-la/runpod-worker-comfy/commit/9170191eccb65de2f17009f68952a18fc008fa6a))

# [3.0.0](https://github.com/blib-la/runpod-worker-comfy/compare/2.2.0...3.0.0) (2024-07-26)

### Features

- support sd3 ([#46](https://github.com/blib-la/runpod-worker-comfy/issues/46)) ([dde69d6](https://github.com/blib-la/runpod-worker-comfy/commit/dde69d6ca75eb7e4c5f01fd17e6da5b62f8a401f))
- provide a base image (#41)

### BREAKING CHANGES

- we have 3 different images now instead of just one:
  - `timpietruskyblibla/runpod-worker-comfy:3.0.0-base`: doesn't contain any checkpoints, just a clean ComfyUI image
  - `timpietruskyblibla/runpod-worker-comfy:3.0.0-sdxl`: contains the checkpoints and VAE for Stable Diffusion XL
    - Checkpoint: [sd_xl_base_1.0.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
    - VAEs:
      - [sdxl_vae.safetensors](https://huggingface.co/stabilityai/sdxl-vae/)
      - [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/)
  - `timpietruskyblibla/runpod-worker-comfy:3.0.0-sd3`: contains the [sd3_medium_incl_clips_t5xxlfp8.safetensors](https://huggingface.co/stabilityai/stable-diffusion-3-medium) checkpoint for Stable Diffusion 3
- `latest` will not be updated anymore
- every branch gets their own 3 images deployed automatically to Docker Hub

# [2.2.0](https://github.com/blib-la/runpod-worker-comfy/compare/2.1.3...2.2.0) (2024-06-04)

### Bug Fixes

- don’t persist credentials ([1546420](https://github.com/blib-la/runpod-worker-comfy/commit/15464201b24de0746fe365e7635540330887a393))
- use custom GITHUB_TOKEN to bypass branch protection ([9b6468a](https://github.com/blib-la/runpod-worker-comfy/commit/9b6468a40b8a476d7812423ff6fe7b73f5f91f1d))

### Features

- network-volume; execution time config; skip default images; access ComfyUI via web ([#35](https://github.com/blib-la/runpod-worker-comfy/issues/35)) ([070cde5](https://github.com/blib-la/runpod-worker-comfy/commit/070cde5460203e24e3fbf68c4ff6c9a9b7910f3f)), closes [#16](https://github.com/blib-la/runpod-worker-comfy/issues/16)

## [2.1.3](https://github.com/blib-la/runpod-worker-comfy/compare/2.1.2...2.1.3) (2024-05-28)

### Bug Fixes

- images in subfolders are not working, fixes [#12](https://github.com/blib-la/runpod-worker-comfy/issues/12) ([37480c2](https://github.com/blib-la/runpod-worker-comfy/commit/37480c2d217698f799f6388ff311b9f8c6c38804))

## [2.1.2](https://github.com/blib-la/runpod-worker-comfy/compare/2.1.1...2.1.2) (2024-05-27)

### Bug Fixes

- removed xl_more_art-full_v1 because civitai requires login now ([2e8e638](https://github.com/blib-la/runpod-worker-comfy/commit/2e8e63801a7672e4923eaad0c18a4b3e2c14d79c))

## [2.1.1](https://github.com/blib-la/runpod-worker-comfy/compare/2.1.0...2.1.1) (2024-05-27)

### Bug Fixes

- check_server default values for delay and check-interval ([4945a9d](https://github.com/blib-la/runpod-worker-comfy/commit/4945a9d65b55aae9117591c8d64f9882d200478e))

# [2.1.0](https://github.com/blib-la/runpod-worker-comfy/compare/2.0.0...2.1.0) (2024-02-12)

### Bug Fixes

- **semantic-release:** added .releaserc ([#21](https://github.com/blib-la/runpod-worker-comfy/issues/21)) ([12b763d](https://github.com/blib-la/runpod-worker-comfy/commit/12b763d8703ce07331a16d4013975f9edc4be3ff))

### Features

- run the worker locally ([#19](https://github.com/blib-la/runpod-worker-comfy/issues/19)) ([34eb32b](https://github.com/blib-la/runpod-worker-comfy/commit/34eb32b72455e6e628849e50405ed172d846d2d9))

# (2023-11-18)

## [1.1.1](https://github.com/blib-la/runpod-worker-comfy/compare/1.1.0...1.1.1) (2023-11-17)

### Bug Fixes

- return the output of "process_output_image" and access jobId correctly ([#11](https://github.com/blib-la/runpod-worker-comfy/issues/11)) ([dc655ea](https://github.com/blib-la/runpod-worker-comfy/commit/dc655ea0dd0b294703f52f6017ce095c3b411527))

# [1.1.0](https://github.com/blib-la/runpod-worker-comfy/compare/1.0.0...1.1.0) (2023-11-17)

### Bug Fixes

- path should be "loras" and not "lora" ([8e579f6](https://github.com/blib-la/runpod-worker-comfy/commit/8e579f63e18851b0be67bff7a42a8e8a46223f2b))

### Features

- added unit tests for everthing, refactored the code to make it better testable, added test images ([a7492ec](https://github.com/blib-la/runpod-worker-comfy/commit/a7492ec8f289fc64b8e54c319f47804c0a15ae54))
- added xl_more_art-full_v1, improved comments ([9aea8ab](https://github.com/blib-la/runpod-worker-comfy/commit/9aea8abe1375f3d48aa9742c444b5242111e3121))
- base64 image output ([#8](https://github.com/blib-la/runpod-worker-comfy/issues/8)) ([76bf0b1](https://github.com/blib-la/runpod-worker-comfy/commit/76bf0b166b992a208c53f5cb98bd20a7e3c7f933))

# [1.0.0](https://github.com/blib-la/runpod-worker-comfy/compare/ecfec1349da0d04ea5f21c82d8903e1a5bd3c923...1.0.0) (2023-10-12)

### Bug Fixes

- don't run ntpdate as this is not working in GitHub Actions ([2f7bd3f](https://github.com/blib-la/runpod-worker-comfy/commit/2f7bd3f71f24dd3b6ecc56f3a4c27bbc2d140eca))
- got rid of syntax error ([c04de4d](https://github.com/blib-la/runpod-worker-comfy/commit/c04de4dea93dbe586a9a887e04907b33597ff73e))
- updated path to "comfyui" ([37f66d0](https://github.com/blib-la/runpod-worker-comfy/commit/37f66d04b8c98810714ffbc761412f3fcdb1d861))

### Features

- added default ComfyUI workflow ([fa6c385](https://github.com/blib-la/runpod-worker-comfy/commit/fa6c385e0dc9487655b42772bb6f3a5f5218864e))
- added runpod as local dependency ([9deae9f](https://github.com/blib-la/runpod-worker-comfy/commit/9deae9f5ec723b93540e6e2deac04b8650cf872a))
- example on how to configure the .env ([4ed5296](https://github.com/blib-la/runpod-worker-comfy/commit/4ed529601394e8a105d171ab1274737392da7df5))
- logs should be written to stdout so that we can see them inside the worker ([fc731ff](https://github.com/blib-la/runpod-worker-comfy/commit/fc731fffcd79af67cf6fcdf6a6d3df6b8e30c7b5))
- simplified input ([35c2341](https://github.com/blib-la/runpod-worker-comfy/commit/35c2341deca346d4e6df82c36e101b7495f3fc03))
- simplified input to just have "prompt", removed unused code ([0c3ccda](https://github.com/blib-la/runpod-worker-comfy/commit/0c3ccda9c5c8cdc56eae829bb358ceb532b36371))
- updated path to "comfyui", added "ntpdate" to have the time of the container in sync with AWS ([2fda578](https://github.com/blib-la/runpod-worker-comfy/commit/2fda578d62460275abec11d6b2fbe5123d621d5f))
- use local ".env" to load env variables, mount "comfyui/output" to localhost so that people can see the generated images ([aa645a2](https://github.com/blib-la/runpod-worker-comfy/commit/aa645a233cd6951d296d68f7ddcf41b14b3f4cf9))
- use models from huggingface, not from local folder ([b1af369](https://github.com/blib-la/runpod-worker-comfy/commit/b1af369bb577c0aaba8875d8b2076e1888356929))
- wait until server is ready, wait until image generation is done, upload to s3 ([ecfec13](https://github.com/blib-la/runpod-worker-comfy/commit/ecfec1349da0d04ea5f21c82d8903e1a5bd3c923))
