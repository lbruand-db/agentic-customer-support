# Databricks notebook source
# DBTITLE 1,Get Workspace Client
from databricks.sdk import WorkspaceClient

workspace_client = WorkspaceClient()

# COMMAND ----------

service_principal = workspace_client.service_principals.create(
    application_id="fc51b87a-e4da-47a9-820e-deb6c056ec8b",
    display_name="telco-customer-support-sp"
)

# COMMAND ----------

telco_customer_support_group = workspace_client.groups.create(
    display_name="telco-customer-support"
)

# COMMAND ----------


gold_schema = workspace_client.schemas.create(
    catalog_name="workspace",
    name="gold",
    comment="Gold schema for curated data"
)


# COMMAND ----------

agent_schema = workspace_client.schemas.create(
    catalog_name="workspace",
    name="agent",
    comment="Agent schema for agents"
)

# COMMAND ----------

from databricks.sdk.service.catalog import VolumeType

tech_support_volume = workspace_client.volumes.create(
    catalog_name="workspace",
    schema_name="gold",
    name="tech_support",
    volume_type=VolumeType.MANAGED,
    comment="Volume for tech support data"
)

# COMMAND ----------

workspace_client.workspace.mkdirs("/Workspace/Shared/telco_support_agent/dev")

# COMMAND ----------

workspace_client.workspace.deploy_bundle(
    bundle_path="."
)
