# Publish MCP servers in the AI Hub

Set `litellm_settings.public_mcp_servers` to the concrete IDs of the servers you want listed in the public AI Hub. Pin `server_id` in each configuration entry so the publication list stays stable across deployments

```yaml
mcp_servers:
  documentation:
    server_id: documentation-mcp
    url: https://mcp.example.com/mcp
    transport: http
    available_on_public_internet: true

litellm_settings:
  public_mcp_hub_strict_whitelist: true
  public_mcp_servers:
    - documentation-mcp
```

Use `documentation-mcp`, the `server_id`, in the publication list. The configuration key `documentation`, display names, and aliases are not publication IDs. Database-created servers use the ID returned by `/v1/mcp/server`

The dashboard's **AI Hub > MCP Hub > Manage MCP Hub Visibility** dialog edits this same list. Its YAML example includes the selected server IDs. With database-backed configuration (`store_model_in_db: true`), a value declared in YAML is owned by that file: edit the file and reload, or remove that key from YAML to let the dashboard manage it in the database. File-backed deployments can save the list directly to their configuration file

To remove all explicit entries, save an empty selection in the dialog or configure:

```yaml
litellm_settings:
  public_mcp_hub_strict_whitelist: true
  public_mcp_servers: []
```

## Hub listing and network access

**Hub: Listed** means the server appears in `/public/mcp_hub`. The dashboard derives this status from the current registry and publication settings. Setting `mcp_info.is_public` on a server does not publish it; that response field is derived metadata. `mcp_info.is_public_explicit` identifies registered servers included in the explicit publication list

**Public IP setting** refers to `available_on_public_internet`, which defaults to `true`. It permits requests from public IP addresses. For compatibility, explicit membership in `public_mcp_servers` also permits public-IP requests even when this per-server setting is disabled. Publication and network settings do not bypass the gateway's authentication or tool permissions

The default `public_mcp_hub_strict_whitelist: true` lists only registered servers in `public_mcp_servers`. Legacy mode (`false`) additionally lists registered servers with `available_on_public_internet: true`. In legacy mode, clearing the explicit publication list leaves these automatically listed servers visible. Enable strict mode when the publication list should fully determine hub visibility
