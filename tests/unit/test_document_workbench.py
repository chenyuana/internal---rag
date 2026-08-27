from fastapi.testclient import TestClient


def test_document_workbench_and_assets_are_served(client: TestClient) -> None:
    interface = client.get("/documents")
    stylesheet = client.get("/assets/documents.css")
    script = client.get("/assets/documents.js")
    outline_parser = client.get("/assets/outline_parser.js")

    assert interface.status_code == 200
    assert 'id="workbench"' in interface.text
    assert 'id="templateFile"' in interface.text
    assert 'id="evidenceFiles"' in interface.text
    assert 'id="generateSection"' in interface.text
    assert "结构文件生成初始大纲" in interface.text
    assert "在智能问答中检索本节证据" not in interface.text
    assert stylesheet.status_code == 200
    assert ".workbench" in stylesheet.text
    assert script.status_code == 200
    assert outline_parser.status_code == 200
    assert "parseOutlineFromText" in outline_parser.text
    assert "parseOutlineSections" in outline_parser.text
    assert "suggestGuidance" in outline_parser.text
    assert "OutlineParser.parseOutlineSections" in script.text
    assert "已从结构文件识别出" in script.text
    assert "/api/v1/documents/sections/guidance-suggest" in script.text
    assert 'id="generateGuidance"' in interface.text
    assert 'id="optimizeGuidance"' in interface.text
    assert "AIRWORTHINESS_OUTLINE" in script.text
    assert "/api/v1/documents/sections/generate" in script.text
    assert "target_length" in script.text
    assert "正文已保留供人工核对" in script.text
    assert "本节引用" in script.text
    assert "data-outline-title" in script.text
    assert "data-delete-section" in script.text
    assert ".outline-title-input" in stylesheet.text
    assert "overscroll-behavior:contain" in stylesheet.text
    assert "scrollbar-gutter:stable" in stylesheet.text
    assert 'id="openModelDialog"' in interface.text
    assert 'id="connectionBaseUrl"' in interface.text
    assert "/api/v1/admin/model-connections" in script.text
    assert 'id="documentType"' in interface.text
    assert 'id="sectionGuidance"' in interface.text
    assert "OUTLINE_PRESETS" in script.text
    assert "TEST_PLAN_OUTLINE" in script.text


def test_existing_pages_link_to_document_workbench(client: TestClient) -> None:
    for path in ("/ingestion", "/chat", "/evaluation"):
        response = client.get(path)

        assert response.status_code == 200
        assert 'href="/documents"' in response.text
