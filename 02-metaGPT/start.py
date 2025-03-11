# %% [markdown]
# # 安装metaGPT
# ```
# pip install metagpt
# ```

# %% [markdown]
# # 配置大模型API
# https://docs.deepwisdom.ai/main/zh/guide/get_started/configuration/llm_api_configuration.html
# 
# 这里演示智谱API，智谱api目前有免费的模型可以调用，但是需要申请账号，申请后，在config/config2.yaml中配置好即可
# 
# 
# 

# %%
import asyncio
from metagpt.roles import (
    Architect,
    Engineer,
    ProductManager,
    ProjectManager,
)
from metagpt.team import Team

# %%
async def startup(idea: str):
    company = Team()
    company.hire(
        [
            ProductManager(),
            Architect(),
            ProjectManager(),
            Engineer(),
        ]
    )
    company.invest(investment=3.0)
    company.run_project(idea=idea)

    await company.run(n_round=5)


# # %%
# history = await startup(idea="write a 2048 game")
asyncio.run(startup(idea="write a 2048 game"))

# %%



