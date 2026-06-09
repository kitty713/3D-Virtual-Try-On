import bpy
import numpy as np
from mathutils import Vector

# =========================
# 用户配置
# =========================
NPZ_PATH = r"G:\smplx_rig_data (1).npz"  # 改成你的 npz 路径
MESH_NAME = "SMPLX_Mesh"
ARM_NAME = "SMPLX_Armature"
BONE_PREFIX = "J_"

# 是否清理同名对象
DELETE_OLD = True

# 坐标轴转换（如果导入后方向不对可尝试打开）
# SMPL-X 常见坐标与 Blender 坐标可能不一致
APPLY_AXIS_CONVERSION = False

def convert_axis(v):
    """可选坐标转换：SMPL(x,y,z) -> Blender(x,-z,y)"""
    if not APPLY_AXIS_CONVERSION:
        return v
    return np.array([v[0], -v[2], v[1]], dtype=np.float32)

# =========================
# 读取数据
# =========================
data = np.load(NPZ_PATH)
required_keys = ["verts", "faces", "joints", "weights", "parents"]
for k in required_keys:
    if k not in data:
        raise KeyError(f"npz 缺少关键字段: {k}")

verts = data["verts"].astype(np.float32)      # (N, 3)
faces = data["faces"].astype(np.int32)        # (F, 3)
joints = data["joints"].astype(np.float32)    # (J, 3)
weights = data["weights"].astype(np.float32)  # (N, J)
parents = data["parents"].astype(np.int32)    # (J,)

N = verts.shape[0]
J = joints.shape[0]

if weights.shape[0] != N:
    raise ValueError(f"weights 顶点数不匹配: {weights.shape[0]} != {N}")
if weights.shape[1] != J:
    raise ValueError(f"weights 关节数不匹配: {weights.shape[1]} != {J}")
if parents.shape[0] != J:
    raise ValueError(f"parents 数量不匹配: {parents.shape[0]} != {J}")

verts = np.array([convert_axis(v) for v in verts], dtype=np.float32)
joints = np.array([convert_axis(j) for j in joints], dtype=np.float32)

# =========================
# 删除旧对象（可选）
# =========================
def remove_object_if_exists(name):
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)

if DELETE_OLD:
    remove_object_if_exists(MESH_NAME)
    remove_object_if_exists(ARM_NAME)

# =========================
# 创建 Mesh
# =========================
mesh_data = bpy.data.meshes.new(MESH_NAME + "_Data")
mesh_obj = bpy.data.objects.new(MESH_NAME, mesh_data)
bpy.context.collection.objects.link(mesh_obj)

mesh_data.from_pydata(verts.tolist(), [], faces.tolist())
mesh_data.update()

# =========================
# 创建 Armature
# =========================
arm_data = bpy.data.armatures.new(ARM_NAME + "_Data")
arm_obj = bpy.data.objects.new(ARM_NAME, arm_data)
bpy.context.collection.objects.link(arm_obj)

# 设为活动对象并进入编辑模式
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode='EDIT')

edit_bones = arm_data.edit_bones
bone_names = [f"{BONE_PREFIX}{i:02d}" for i in range(J)]

# 先建所有骨头（head/tail）
# tail 规则：指向子骨平均方向；若无子骨则给一个小偏移
children = [[] for _ in range(J)]
for i, p in enumerate(parents):
    if 0 <= p < J and p != i:
        children[p].append(i)

for i in range(J):
    b = edit_bones.new(bone_names[i])
    head = Vector(joints[i].tolist())
    b.head = head

    if len(children[i]) > 0:
        child_pos = np.mean(joints[children[i]], axis=0)
        tail = Vector(child_pos.tolist())
        # 避免 head/tail 重合
        if (tail - head).length < 1e-5:
            tail = head + Vector((0.0, 0.03, 0.0))
    else:
        # 末端骨给一个小尾巴
        b.parent = None
        b.use_connect = False
        b.roll = 0.0
        # 根据父骨方向估计末端方向
        p = parents[i]
        if 0 <= p < J and p != i:
            direction = joints[i] - joints[p]
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                direction = np.array([0.0, 0.03, 0.0], dtype=np.float32)
            else:
                direction = direction / norm * 0.03
            tail = head + Vector(direction.tolist())
        else:
            tail = head + Vector((0.0, 0.03, 0.0))

    b.tail = tail

# 再设置父子关系
for i in range(J):
    p = int(parents[i])
    if 0 <= p < J and p != i:
        edit_bones[bone_names[i]].parent = edit_bones[bone_names[p]]

# 返回对象模式
bpy.ops.object.mode_set(mode='OBJECT')

# =========================
# 创建顶点组并写入权重
# =========================
# 先创建所有骨骼对应的顶点组
for i in range(J):
    mesh_obj.vertex_groups.new(name=bone_names[i])

# 给每个顶点分配权重（过滤非常小的权重）
threshold = 1e-6
for vid in range(N):
    ws = weights[vid]
    nz = np.where(ws > threshold)[0]
    for j in nz:
        w = float(ws[j])
        mesh_obj.vertex_groups[bone_names[j]].add([vid], w, 'REPLACE')

# 规范化（防止总和偏离1）
# Blender 会在很多情况下自动处理，这里手动确保更稳
for v in mesh_obj.data.vertices:
    total = 0.0
    groups = v.groups
    for g in groups:
        total += g.weight
    if total > 1e-8:
        for g in groups:
            g.weight = g.weight / total

# =========================
# 绑定 Armature Modifier
# =========================
mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
mod.object = arm_obj
mod.use_vertex_groups = True

# 可选：显示在前，方便选骨骼
arm_obj.show_in_front = True

# 选中骨架方便直接进 Pose
bpy.context.view_layer.objects.active = arm_obj
arm_obj.select_set(True)
mesh_obj.select_set(False)

print("完成：SMPL-X 网格与骨架绑定成功。")
print(f"Mesh: {MESH_NAME}, Armature: {ARM_NAME}, Joints: {J}, Verts: {N}")


import bpy

# ==============================
# 用户配置（按需修改）
# ==============================
BODY_NAME = "SMPLX_Mesh"         # 人体mesh对象名
ARMATURE_NAME = "SMPLX_Armature" # 骨架对象名
CLOTH_PREFIX = "bluedress"          # 可选：只处理名字以此前缀开头的衣服；设为 "" 表示不过滤
ADD_SHRINKWRAP = True           # 是否给衣服加轻微Shrinkwrap防穿模
SHRINKWRAP_OFFSET = 0.003        # 米
CLEAR_OLD_GROUPS = False         # 是否清空衣服已有顶点组（谨慎）
MAX_DISTANCE = 0.05              # Data Transfer最大距离（米），可按模型尺寸调
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