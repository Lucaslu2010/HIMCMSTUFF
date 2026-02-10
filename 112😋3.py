def total_cost_with_positions(params):
    """给定料场位置，计算最小总成本"""
    # params: [x1, y1, x2, y2]
    depot_positions = [params[0:2], params[2:4]]

    # 计算距离矩阵
    dist = np.zeros((n_sites, n_depots))
    for i in range(n_sites):
        for j in range(n_depots):
            pos = np.array([sites['a'][i], sites['b'][i]])
            dist[i, j] = np.linalg.norm(pos - depot_positions[j])

    # 求解运输问题（给定距离）
    c = dist.flatten()

    # 约束矩阵（与之前相同）
    A_eq = []
    for i in range(n_sites):
        row = np.zeros(n_sites * n_depots)
        row[i * n_depots:(i + 1) * n_depots] = 1
        A_eq.append(row)

    A_ub = []
    for j in range(n_depots):
        row = np.zeros(n_sites * n_depots)
        for i in range(n_sites):
            row[i * n_depots + j] = 1
        A_ub.append(row)

    b_eq = sites['d']
    b_ub = [capacity, capacity]
    bounds = [(0, None)] * (n_sites * n_depots)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        return result.fun
    else:
        return 1e10  # 返回一个大数表示不可行


# 初始猜测：原来的料场位置
initial_guess = [depots['A'][0], depots['A'][1], depots['B'][0], depots['B'][1]]

# 优化料场位置
bounds = [(0, 10), (0, 10), (0, 10), (0, 10)]  # 合理的位置范围

result = minimize(total_cost_with_positions, initial_guess,
                  method='L-BFGS-B', bounds=bounds,
                  options={'maxiter': 1000, 'disp': True})

if result.success:
    optimal_positions = result.x.reshape(2, 2)
    optimal_cost = result.fun

    print("\n" + "=" * 60)
    print("子问题（2）结果：优化料场位置")
    print("=" * 60)
    print(f"新料场1位置: ({optimal_positions[0, 0]:.4f}, {optimal_positions[0, 1]:.4f})")
    print(f"新料场2位置: ({optimal_positions[1, 0]:.4f}, {optimal_positions[1, 1]:.4f})")
    print(f"最小总吨千米数: {optimal_cost:.4f}")
    print(f"相比原方案节省: {cost_fixed - optimal_cost:.4f} 吨千米")
    print(f"节省百分比: {(cost_fixed - optimal_cost) / cost_fixed * 100:.2f}%")

    # 获取最优配送方案
    x_opt_new, _, _ = solve_transport_problem([optimal_positions[0], optimal_positions[1]])

    print("\n新配送方案（吨）:")
    print("工地 | 从新料场1 | 从新料场2 | 总需求")
    print("-" * 45)
    for i in range(n_sites):
        print(f"{i + 1:2d}   | {x_opt_new[i, 0]:9.3f} | {x_opt_new[i, 1]:9.3f} | {sites['d'][i]:5.1f}")
else:
    print("优化失败")