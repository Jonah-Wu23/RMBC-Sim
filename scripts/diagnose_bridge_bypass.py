#!/usr/bin/env python
"""
C-lite 诊断脚本：找到 passenger 可达但 bus 不可达的边
目标：识别 142955→97070 走廊上需要放行 bus 权限的最小边集
"""
import sumolib

def find_bus_blockers(net, from_id, to_id):
    """
    找到从 from_id 到 to_id 的 passenger 路径，
    然后识别该路径上阻止 bus 的边
    """
    print(f"\n{'='*60}")
    print(f"分析走廊: {from_id} → {to_id}")
    
    if not net.hasEdge(from_id) or not net.hasEdge(to_id):
        print("  错误: 边不存在")
        return []

    start = net.getEdge(from_id)
    end = net.getEdge(to_id)
    
    # Step 1: 找 passenger 可达路径
    print("\n[Step 1] 查找 passenger 可达路径...")
    path_passenger, cost = net.getOptimalPath(start, end, vClass='passenger')
    
    if not path_passenger:
        print("  ❌ 连 passenger 都无法到达！网络结构严重断裂")
        return []
    
    print(f"  ✅ 找到 passenger 路径: {len(path_passenger)} 条边, 总成本: {cost:.2f}")
    
    # Step 2: 查找 bus 路径
    print("\n[Step 2] 查找 bus 可达路径...")
    path_bus, bus_cost = net.getOptimalPath(start, end, vClass='bus')
    
    if path_bus:
        print(f"  ✅ bus 也可达! {len(path_bus)} 条边, 成本: {bus_cost:.2f}")
        print("  → 无需修复，bus 已经可以通行")
        return []
    else:
        print("  ❌ bus 无法到达 - 需要识别阻塞点")
    
    # Step 3: 沿 passenger 路径找出阻止 bus 的边
    print("\n[Step 3] 识别 passenger 路径上阻止 bus 的边...")
    blockers = []
    
    for edge in path_passenger:
        lane = edge.getLane(0)
        perms = lane.getPermissions()
        edge_id = edge.getID()
        
        if 'bus' not in perms:
            blockers.append({
                'edge_id': edge_id,
                'length': edge.getLength(),
                'permissions': perms,
            })
            print(f"  🚫 {edge_id}: 长度={edge.getLength():.1f}m, 当前权限={perms}")
    
    if not blockers:
        print("  所有边都允许 bus，问题可能在 connection 层级")
        # 检查 connection
        for i in range(len(path_passenger) - 1):
            from_edge = path_passenger[i]
            to_edge = path_passenger[i + 1]
            
            # 获取连接
            outgoing = from_edge.getOutgoing()
            if to_edge in outgoing:
                for conn in outgoing[to_edge]:
                    # sumolib Connection 可能有 getVClass 或类似方法
                    # 这里简化处理，假设 edge-level 权限是主要问题
                    pass
    
    print(f"\n[总结] 找到 {len(blockers)} 个阻塞 bus 的边")
    return blockers


def main():
    net_file = 'sumo/net/hk_irn_v3_patched_v1.net.xml'
    print(f"加载网络: {net_file}")
    net = sumolib.net.readNet(net_file)
    
    # 分析 GAP_7 对应的断点: 142955 → 97070
    blockers = find_bus_blockers(net, '142955', '97070')
    
    if blockers:
        print("\n" + "="*60)
        print("需要修复的边（为其添加 bus 权限）:")
        print("="*60)
        for b in blockers:
            print(f"  - {b['edge_id']}")
        
        # 输出修复建议
        print("\n修复方案:")
        print("在 tmp/hk_v3_plain.edg.xml 中为以下边添加 allow='bus' 或移除 disallow='bus':")
        for b in blockers:
            print(f"  {b['edge_id']}")

if __name__ == "__main__":
    main()
