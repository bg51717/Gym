# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This responses_api_models server is only used to proxy to one or more existing LocalVLLMModel servers so we don't need to duplicate GPU resources.

`model_server` accepts a single ref or a list of refs (mirroring `base_url`'s str-or-list
convention). With several refs, the proxy aggregates every upstream's inner vLLM URL and the
inherited VLLMModel client round-robins new sessions across them — N independent single-node
instances behave as one logical model server.
"""

from time import sleep
from typing import List, Union

import requests
from pydantic import Field, model_validator

from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class LocalVLLMModelProxyServerConfig(VLLMModelConfig):
    # We inherit these configs from VLLMModelConfig, but they are set to optional since we will get this information after the referenced LocalVLLMModel spinup
    base_url: Union[str, List[str]] = Field(default_factory=list)
    # Not used on local deployments
    api_key: str = "dummy"  # pragma: allowlist secret
    model: str = "dummy"

    # A single upstream ref, or a list of them to fan out across several upstream
    # LocalVLLMModel servers (all must serve the same model). Normalized to a list.
    model_server: Union[ModelServerRef, List[ModelServerRef]]

    @model_validator(mode="after")
    def _normalize_model_server(self) -> "LocalVLLMModelProxyServerConfig":
        if isinstance(self.model_server, ModelServerRef):
            self.model_server = [self.model_server]
        if not self.model_server:
            raise ValueError("`model_server` must contain at least one ref.")
        return self


class LocalVLLMModelProxyServer(VLLMModel):
    config: LocalVLLMModelProxyServerConfig

    def setup_webserver(self):
        base_urls: List[str] = []
        for ref in self.config.model_server:
            model_server_name = ref.name

            print(f"Waiting for LocalVLLMModelServer `{model_server_name}` spinup")

            while self.server_client.poll_for_status(model_server_name) != "success":
                # Sleep for 10s by default
                sleep(10)

            model_server_config_dict = get_first_server_config_dict(
                self.server_client.global_config_dict, model_server_name
            )
            model_server_base_url = self.server_client._build_server_base_url(model_server_config_dict)
            response = requests.get(
                f"{model_server_base_url}/get_inner_vllm_config",
            )
            assert response.ok

            response_dict = response.json()

            inner_base_url = response_dict["base_url"]
            base_urls.extend(inner_base_url if isinstance(inner_base_url, list) else [inner_base_url])
            self.config.api_key = response_dict["api_key"]
            assert self.config.model in ("dummy", response_dict["model"]), (
                f"All upstream servers must serve the same model; got `{response_dict['model']}` "
                f"from `{model_server_name}` after `{self.config.model}`"
            )
            self.config.model = response_dict["model"]

        self.config.base_url = base_urls

        # Reset clients after base_url config
        self._post_init()

        return super().setup_webserver()


if __name__ == "__main__":
    LocalVLLMModelProxyServer.run_webserver()
