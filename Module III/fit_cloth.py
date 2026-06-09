import bpy

# ==============================
# 用户配置（按需修改）
# ==============================
BODY_NAME = "SMPLX_Mesh"         # 人体mesh对象名
ARMATURE_NAME = "SMPLX_Armature" # 骨架对象名
CLOTH_PREFIX = "blackdress1"          # 可选：只处理名字以此前缀开头的衣服；设为 "" 表示不过滤
ADD_SHRINKWRAP = True           # 是否给衣服加轻微Shrinkwrap防穿模
SHRINKWRAP_OFFSET = 0.010        # 米
CLEAR_OLD_GROUPS = False         # 是否清空衣服已有顶点组（谨慎）
MAX_DISTANCE = 0.25              # Data Transfer最大距离（米），可按模型尺寸调
MIX_MODE = 'REPLACE'             # 'REPLACE' 或 'ADD'
VERT_MAPPING = 'POLYINTERP_NEAREST'  # 推荐最近面插值

# ==============================
# 基础检查
# ==============================
body = bpy.data.objects.get(BODY_NAME)
arm = bpy.data.objects.get(ARMATURE_NAME)

if body is None:
    raise ValueError(f"找不到人体对象: {BODY_NAME}")
if arm is None or arm.type != 'ARMATURE':
    raise ValueError(f"找不到骨架对象或类型不对: {ARMATURE_NAME}")

if body.type != 'MESH':
    raise ValueError(f"{BODY_NAME} 不是 MESH 对象")

# 候选衣服：场景中所有Mesh，排除人体本体
cloth_objs = []
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH':
        continue
    if obj.name == BODY_NAME:
        continue
    if CLOTH_PREFIX and not obj.name.startswith(CLOTH_PREFIX):
        continue
    cloth_objs.append(obj)

if not cloth_objs:
    raise ValueError("没有找到可处理的衣服对象。检查 CLOTH_PREFIX 或对象类型。")

# ==============================
# 工具函数
# ==============================
def ensure_armature_modifier(obj, armature_obj):
    mod = None
    for m in obj.modifiers:
        if m.type == 'ARMATURE':
            mod = m
            break
    if mod is None:
        mod = obj.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = armature_obj
    mod.use_vertex_groups = True
    return mod

def move_modifier_to_top(obj, mod_name):
    # 尝试把Armature放到较前位置，减少顺序引起的问题
    idx = obj.modifiers.find(mod_name)
    if idx <= 0:
        return
    bpy.context.view_layer.objects.active = obj
    for _ in range(idx):
        try:
            bpy.ops.object.modifier_move_up(modifier=mod_name)
        except RuntimeError:
            break

def remove_all_vertex_groups(obj):
    while obj.vertex_groups:
        obj.vertex_groups.remove(obj.vertex_groups[0])

def add_or_configure_shrinkwrap(obj, target, offset=0.003):
    sw = None
    for m in obj.modifiers:
        if m.type == 'SHRINKWRAP':
            sw = m
            break
    if sw is None:
        sw = obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    sw.target = target
    sw.wrap_method = 'NEAREST_SURFACEPOINT'
    sw.wrap_mode = 'OUTSIDE'
    sw.offset = offset
    return sw

def transfer_weights(src_body, dst_cloth):
    # 选中并激活目标
    bpy.ops.object.select_all(action='DESELECT')
    src_body.select_set(True)
    dst_cloth.select_set(True)
    bpy.context.view_layer.objects.active = dst_cloth

    # 可选清空旧组
    if CLEAR_OLD_GROUPS:
        remove_all_vertex_groups(dst_cloth)

    # Data Transfer: 顶点组权重
    dt = None
    for m in dst_cloth.modifiers:
        if m.type == 'DATA_TRANSFER' and m.name == "DT_WeightsFromBody":
            dt = m
            break
    if dt is None:
        dt = dst_cloth.modifiers.new(name="DT_WeightsFromBody", type='DATA_TRANSFER')

    dt.object = src_body
    dt.use_vert_data = True
    dt.data_types_verts = {'VGROUP_WEIGHTS'}
    dt.vert_mapping = VERT_MAPPING
    dt.use_max_distance = True
    dt.max_distance = MAX_DISTANCE
    dt.mix_mode = MIX_MODE
    dt.mix_factor = 1.0

    # 应用Data Transfer（写入顶点组）
    bpy.context.view_layer.objects.active = dst_cloth
    bpy.ops.object.modifier_apply(modifier=dt.name)

# ==============================
# 执行批处理
# ==============================
processed = []
failed = []

for cloth in cloth_objs:
    try:
        # 1) 转权重
        transfer_weights(body, cloth)

        # 2) 绑定同一骨架
        arm_mod = ensure_armature_modifier(cloth, arm)
        move_modifier_to_top(cloth, arm_mod.name)

        # 3) 可选防穿模
        if ADD_SHRINKWRAP:
            add_or_configure_shrinkwrap(cloth, body, SHrinkwrap_OFFSET if 'SHrinkwrap_OFFSET' in globals() else SHRINKWRAP_OFFSET)

        processed.append(cloth.name)
    except Exception as e:
        failed.append((cloth.name, str(e)))

# ==============================
# 输出结果
# ==============================
print("=" * 50)
print(f"完成。成功处理: {len(processed)}")
for n in processed:
    print(f"  [OK] {n}")

if failed:
    print(f"失败: {len(failed)}")
    for n, err in failed:
        print(f"  [FAIL] {n}: {err}")
print("=" * 50)