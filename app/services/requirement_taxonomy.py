from __future__ import annotations

import re

from app.schemas.evidence import RequirementType

# 迁移期领域词表：只用于召回与旧数据缺少 requirement_type 时的兼容判定。
# Stage 6 是否删除由 retrieval_cases 的有/无词表 Recall@8 对照决定。
ASPECT_RETRIEVAL_TERMS: dict[str, str] = {
    "技术参数": "播种量 航高 速度 种子处理 整地 作业设计",
    "运营管理": "管理制度 服务要求 现场勘察 设备匹配 制定方案 服务实施 成果提交",
    "作业航高": "航高 速度",
    "播种量计算逻辑": "播种量",
    "地面分辨率": "地面分辨率 表1 比例尺",
    "地面分辨率要求": "地面分辨率 表1 比例尺",
    "设备选型": "设备 无人机 相机 载荷 传感器 云台 像素 分辨率 存储 镜头 选型 要求",
    "像控布设": "像控点 控制点 布设 测量 像控",
    "成果验收": (
        "成果验收 成果质量 质量检查 精度检查 限差 中误差 精度 "
        "分辨率 密度 平面位置 高程位置 相对位置 验收 检查记录 "
        "检查报告 验收报告"
    ),
    "成果交付": "成果交付 成果提交 成果清单 巡检成果 巡检报告 影像资料 技术报告",
}

# Broad cells need an independent semantic retrieval view.  These terms are
# issued as a separate query and fused with the untouched original query;
# they are never concatenated into one dense query.  The vocabulary describes
# reusable requirement concepts and deliberately contains no table numbers.
ASPECT_SECONDARY_RETRIEVAL_TERMS: dict[str, str] = {
    "成果验收": (
        "成果质量控制 质量检查 精度限差 中误差 点云密度 点云高程 "
        "地面分辨率 平面位置 高程精度"
    ),
}

SUBJECT_RETRIEVAL_TERMS: dict[str, str] = {
    "水稻植保无人机": "水稻 病虫害 施药 稻纵卷叶螟",
    "林地病虫害喷洒无人机": "林业 有害生物 喷洒 防治",
}

ASPECT_REQUIREMENT_TYPES: tuple[tuple[tuple[str, ...], RequirementType], ...] = (
    (("设备", "选型", "相机", "载荷"), "equipment"),
    (("像控", "控制点", "布设"), "control_point_layout"),
    (("飞行", "航线", "航高"), "flight_operation"),
    (("处理", "建模"), "data_processing"),
    (("质量", "精度", "检查"), "quality_check"),
    (("交付", "提交", "成果清单"), "deliverable"),
    (("验收",), "acceptance"),
    (("报告",), "reporting"),
    (("安全",), "safety"),
    (("管理", "运维"), "management"),
)

# 证据行的类型优先级。一个设备格可以同时召回设备、作业、前资料和
# 安全条款，只有第一类才能作为设备要求。
LINE_REQUIREMENT_MARKERS: tuple[tuple[tuple[str, ...], RequirementType], ...] = (
    (("像控点", "像控", "控制点", "区域网", "空中三角", "航带法"), "control_point_layout"),
    (("验收", "验收标准", "验收要求"), "acceptance"),
    (
        ("成果提交", "成果清单", "成果资料", "成果应", "提交成果", "影像资料", "巡检报告"),
        "deliverable",
    ),
    (
        (
            "质量控制",
            "质量检查",
            "精度检查",
            "限差",
            "精度",
            "检查记录",
        ),
        "quality_check",
    ),
    (("报告", "报告应", "报告包括"), "reporting"),
    (
        (
            "相机",
            "镜头",
            "云台",
            "载荷",
            "传感器",
            "多旋翼",
            "固定翼",
            "无人机",
            "设备选型",
            "分辨率",
            "像素",
        ),
        "equipment",
    ),
    (("航线", "航高", "飞行", "巡检距离", "净空"), "flight_operation"),
    (("安全", "应急", "禁飞"), "safety"),
    (("建模", "空三", "数据处理", "病害识别"), "data_processing"),
)

ASPECT_ALLOWED_REQUIREMENT_TYPES: dict[str, frozenset[RequirementType]] = {
    "设备选型": frozenset({"equipment"}),
    "像控布设": frozenset({"control_point_layout"}),
    # “成果验收”回答合格判据、质量和精度要求；成果清单、技术报告属于
    # “成果交付”，不能因为同处成果章节就把一个验收格判为 covered。
    "成果验收": frozenset({"quality_check", "acceptance"}),
    "成果交付": frozenset({"deliverable", "reporting"}),
}


def requirement_type_for(aspect: str) -> RequirementType:
    for markers, requirement_type in ASPECT_REQUIREMENT_TYPES:
        if any(marker in aspect for marker in markers):
            return requirement_type
    return "unknown"


def requirement_type_for_line(line: str) -> RequirementType:
    """Classify one evidence sentence, rather than inheriting the cell aspect."""
    # Accuracy/limit tables may mention control points as the measured object;
    # their requirement type is still quality_check, not layout.
    if any(
        marker in line
        for marker in (
            "成果质量",
            "成果精度",
            "成果检查",
            "检查点",
            "限差",
            "中误差",
            "点云密度",
            "点云高程精度",
            "地面分辨率",
            "平面位置精度",
            "高程精度",
        )
    ):
        return "quality_check"
    if any(
        marker in line
        for marker in ("像控点", "像控", "控制点", "区域网", "空中三角", "航带法")
    ):
        return "control_point_layout"
    if any(marker in line for marker in ("航线", "航高", "净空", "巡检距离")):
        return "flight_operation"
    if any(
        marker in line
        for marker in ("成果提交", "成果清单", "成果资料", "提交成果", "影像资料", "巡检报告")
    ):
        return "deliverable"
    if any(
        marker in line
        for marker in (
            "质量控制",
            "质量检查",
            "成果质量",
            "成果精度",
            "成果检查",
            "精度检查",
            "精度检查记录表",
            "成果精度检查",
            "限差",
            "中误差",
            "地面分辨率",
            "点云密度",
            "点云高程精度",
            "平面位置精度",
            "高程精度",
        )
    ):
        return "quality_check"
    if any(
        marker in line
        for marker in ("相机", "镜头", "云台", "载荷", "传感器", "多旋翼", "固定翼", "无人机")
    ):
        return "equipment"
    if "设备选型" in line or (
        any(marker in line for marker in ("选择", "选用", "采用"))
        and any(marker in line for marker in ("无人机", "相机", "云台", "载荷", "传感器"))
    ):
        return "equipment"
    for markers, requirement_type in LINE_REQUIREMENT_MARKERS:
        if any(marker in line for marker in markers):
            return requirement_type
    return "unknown"


def line_matches_aspect(aspect: str, line: str) -> bool:
    """Return whether a line is substantive evidence for an aspect.

    Unknown aspects retain the legacy lexical behavior.  Known comparison
    aspects are intentionally strict: a generic retrieval hit must not make a
    cell covered merely because it shares words such as ``无人机`` or ``成果``.
    """
    # These common comparison dimensions need concrete values or evaluation
    # methods.  A sentence which merely names the dimension (for example
    # ``报告包括……精度评价``) is not an answer to that dimension.
    if "成像波段" in aspect or aspect == "波段":
        return bool(
            re.search(r"\d+(?:\.\d+)?\s*(?:nm|μm|um)\b", line, re.IGNORECASE)
            or any(
                marker in line
                for marker in (
                    "可见光波段", "绿光波段", "红光波段", "蓝光波段",
                    "近红外", "红边波段", "多光谱成像",
                )
            )
        )
    if "最佳时段" in aspect or "最佳时间" in aspect:
        return bool(
            re.search(r"\d{1,2}\s*(?:[:：时点])", line)
            or any(
                marker in line
                for marker in (
                    "上午", "下午", "正午", "日出", "日落", "太阳高度角",
                    "光照", "阴影", "无云", "晴朗", "时间段", "时段内",
                )
            )
        )
    if "精度评价指标" in aspect or "精度指标" in aspect:
        return bool(
            re.search(
                r"(?:R\s*[²2]|RMSE|MAE|MAPE|mAP|准确率|正确率|召回率|"
                r"精确率|中误差|均方根误差|相对误差|决定系数|一致性系数)",
                line,
                re.IGNORECASE,
            )
            or re.search(r"(?:误差|精度|准确率)\s*(?:≤|≥|<|>|不大于|不小于)\s*\d", line)
        )

    allowed = ASPECT_ALLOWED_REQUIREMENT_TYPES.get(aspect)
    if allowed is None:
        return True
    requirement_type = requirement_type_for_line(line)
    if aspect == "设备选型" and requirement_type == "equipment":
        if any(
            marker in line
            for marker in ("检查校正", "固定于飞行平台", "连接线", "擦拭", "通电测试")
        ):
            return False
        explicit_choice = False
        if "选择" in line:
            choice_object = line.split("选择", 1)[1]
            for delimiter in ("，", ",", "。", "；", ";"):
                choice_object = choice_object.split(delimiter, 1)[0]
            choice_object = choice_object[:16]
            explicit_choice = any(
                marker in choice_object
                for marker in ("无人机", "多旋翼", "固定翼", "相机", "云台", "传感器", "载荷")
            )
        return explicit_choice or any(
            marker in line
            for marker in (
                "设备选型",
                "选用",
                "采用",
                "配置",
                "具备",
                "搭载",
                "型号",
                "像素",
                "分辨率",
                "焦距",
                "载荷",
                "续航",
                "避障",
                "存储",
                "传感器尺寸",
                "控制精度",
            )
        )
    if aspect == "像控布设" and requirement_type == "control_point_layout":
        return any(
            marker in line
            for marker in (
                "布设",
                "布点",
                "点位",
                "区域网",
                "航带法",
                "平高点",
                "基线",
                "成图区域",
                "数量",
                "位置",
            )
        )
    if aspect == "成果验收" and requirement_type == "quality_check":
        # “精度”本身也可能描述算法训练（MAP）、设备控制或过程调参。
        # 成果验收所需的是成果/检查对象及其合格判据，而非任意质量词。
        return any(
            marker in line
            for marker in (
                "成果质量",
                "成果精度",
                "成果检查",
                "检查点",
                "点云密度",
                "点云高程",
                "平面位置",
                "高程限差",
                "相对位置",
                "限差",
                "中误差",
                "超限",
                "质量检查",
            )
        )
    return requirement_type in allowed
