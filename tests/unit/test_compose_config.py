from __future__ import annotations

from pathlib import Path

import yaml


class ComposeLoader(yaml.SafeLoader):
    pass


def _construct_override(
    loader: ComposeLoader,
    node: yaml.Node,
) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_object(node)


ComposeLoader.add_constructor("!override", _construct_override)


def test_all_persistent_compose_volumes_use_d_drive_root_variable() -> None:
    compose_path = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    volumes = [
        volume for service in compose["services"].values() for volume in service.get("volumes", [])
    ]

    assert volumes
    assert all("RAG_ROOT" in volume for volume in volumes)


def test_ragflow_overlay_uses_named_data_volumes_and_loopback_ports() -> None:
    project_root = Path(__file__).resolve().parents[2]
    overlay_path = project_root / "deploy" / "ragflow" / "docker-compose.d-drive.yml"
    compose = yaml.load(
        overlay_path.read_text(encoding="utf-8"),
        Loader=ComposeLoader,
    )
    services = compose["services"]
    volumes = [volume for service in services.values() for volume in service.get("volumes", [])]
    ports = [port for service in services.values() for port in service.get("ports", [])]
    allowed_read_only_mounts = {
        "D:/internal-rag/runtime/ragflow-0.26.4/rag/llm/chat_model.py:"
        "/ragflow/rag/llm/chat_model.py:ro",
        "D:/internal-rag/runtime/ragflow-0.26.4/api/apps/restful_apis/chunk_api.py:"
        "/ragflow/api/apps/restful_apis/chunk_api.py:ro",
    }

    assert volumes
    assert all(
        volume.startswith("ragflow_")
        or "RAGFLOW_DOCKER_DIR" in volume
        or volume in allowed_read_only_mounts
        for volume in volumes
    )
    assert compose["volumes"]
    assert all(
        volume["name"].startswith("internal-rag-ragflow-")
        for volume in compose["volumes"].values()
    )
    assert ports
    assert all(port.startswith("127.0.0.1:") for port in ports)


def test_windows_docker_installer_uses_hyper_v_and_d_drive() -> None:
    project_root = Path(__file__).resolve().parents[2]
    installer_path = project_root / "deploy" / "docker" / "install-docker-desktop.ps1"
    installer = installer_path.read_text(encoding="utf-8").lower()

    assert "https://desktop.docker.com/" in installer
    assert "--backend=hyper-v" in installer
    assert "--hyper-v-default-data-root=" in installer
    assert "--wsl-default-data-root=" not in installer
    assert "d:\\internal-rag" in installer
