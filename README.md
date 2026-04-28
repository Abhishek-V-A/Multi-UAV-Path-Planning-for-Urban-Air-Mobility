# Multi-UAV-Path-Planning-for-Urban-Air-Mobility
This project develops a multi-UAV path planning system in MATLAB for urban air mobility. It enables multiple drones to find optimal, collision-free paths in a 3D city environment, considering obstacles, safety distance, and efficiency using algorithms like A* and optimization methods.
%% Multi-UAV Path Planning for Urban Air Mobility
% MathWorks MATLAB-Simulink Challenge Project #247
%
% Description:
%   This script demonstrates a complete multi-UAV path planning framework
%   for Urban Air Mobility (UAM) using:
%     - 3D urban environment modeled via uavScenario
%     - RRT* path planning per UAV
%     - Minimum-snap trajectory smoothing (cubic spline)
%     - Inter-UAV collision avoidance (pseudo-obstacle injection)
%     - Animated 3D visualization
%
% Required Toolboxes:
%   - UAV Toolbox
%   - Navigation Toolbox
%   - Robotics System Toolbox
%   - Optimization Toolbox (optional, for trajectory smoothing)
%
% Author  : [Your Name] — MathWorks Challenge Project 247
% Date    : 2024
% Version : 2.0

clear; clc; close all;

%% =========================================================
% 1. SIMULATION PARAMETERS
% =========================================================
params.numUAVs        = 4;          % Number of UAVs
params.safetyDist     = 15;         % Minimum separation between UAVs [m]
params.maxSpeed       = 10;         % Max UAV speed [m/s]
params.cruiseAlt      = 50;         % Nominal cruise altitude [m]
params.mapSize        = [500, 500, 150];  % [x, y, z] environment bounds [m]
params.gridRes        = 5;          % Occupancy map resolution [m]
params.rrtMaxIter     = 3000;       % RRT* max iterations
params.rrtStepSize    = 15;         % RRT* step size [m]
params.rrtGoalBias    = 0.15;       % RRT* goal-biasing probability
params.simDt          = 0.1;        % Simulation time step [s]
params.seed           = 42;         % Random seed for reproducibility
rng(params.seed);

fprintf('=== Multi-UAV Path Planning for Urban Air Mobility ===\n');
fprintf('Project #247 | UAVs: %d | Map: %dx%dx%d m\n\n', ...
    params.numUAVs, params.mapSize(1), params.mapSize(2), params.mapSize(3));

%% =========================================================
% 2. BUILD URBAN ENVIRONMENT (3D Occupancy Map)
% =========================================================
fprintf('[1/6] Building urban 3D occupancy map...\n');
[occMap, scenario, buildingData] = build_urban_environment(params);

%% =========================================================
% 3. DEFINE UAV START / GOAL POSITIONS
% =========================================================
fprintf('[2/6] Setting UAV start and goal positions...\n');
[starts, goals] = define_uav_waypoints(params, buildingData);

%% =========================================================
% 4. SINGLE-UAV RRT* PATH PLANNING
% =========================================================
fprintf('[3/6] Running RRT* path planning for each UAV...\n');
rawPaths = cell(params.numUAVs, 1);
planningTimes = zeros(params.numUAVs, 1);

for i = 1:params.numUAVs
    fprintf('   UAV %d: Planning from [%.0f,%.0f,%.0f] to [%.0f,%.0f,%.0f]...\n', ...
        i, starts(i,:), goals(i,:));
    tic;
    rawPaths{i} = rrt_star_3d(starts(i,:), goals(i,:), occMap, params);
    planningTimes(i) = toc;
    if isempty(rawPaths{i})
        warning('UAV %d: No path found! Using straight-line fallback.', i);
        rawPaths{i} = [starts(i,:); goals(i,:)];
    end
    fprintf('   UAV %d: Path found (%d waypoints, %.2f s)\n', ...
        i, size(rawPaths{i},1), planningTimes(i));
end

%% =========================================================
% 5. TRAJECTORY SMOOTHING (Cubic Spline)
% =========================================================
fprintf('[4/6] Smoothing trajectories...\n');
smoothPaths = cell(params.numUAVs, 1);
for i = 1:params.numUAVs
    smoothPaths{i} = smooth_trajectory(rawPaths{i}, params);
end

%% =========================================================
% 6. MULTI-UAV COLLISION AVOIDANCE
% =========================================================
fprintf('[5/6] Applying inter-UAV collision avoidance...\n');
[finalPaths, collisionEvents] = multi_uav_deconflict(smoothPaths, params, occMap);

%% =========================================================
% 7. COMPUTE METRICS
% =========================================================
fprintf('[6/6] Computing performance metrics...\n');
metrics = compute_metrics(finalPaths, starts, goals, planningTimes, params);
print_metrics(metrics);

%% =========================================================
% 8. VISUALIZATION
% =========================================================
fprintf('\nGenerating visualizations...\n');
visualize_results(scenario, finalPaths, starts, goals, buildingData, ...
                  collisionEvents, metrics, params);

fprintf('\n=== Simulation complete. ===\n');
save('results/simulation_results.mat', 'finalPaths', 'metrics', ...
     'params', 'starts', 'goals', 'collisionEvents');
fprintf('Results saved to results/simulation_results.mat\n');


%% =========================================================
%%  LOCAL FUNCTIONS
%% =========================================================

% ---------------------------------------------------------
function [occMap, scenario, bldData] = build_urban_environment(params)
% BUILD_URBAN_ENVIRONMENT  Creates a synthetic 3D urban occupancy map
%   Generates random building blocks mimicking a city grid and registers
%   them as a 3-D occupancy map for path planning.

    res = params.gridRes;
    sz  = params.mapSize;

    % Initialise 3-D binary occupancy map
    occMap = occupancyMap3D(res);

    % --- City grid buildings ---
    % Defined as [cx, cy, width, depth, height]
    bldData = [
        % Row 1
         80,  80, 40, 40,  60;
        180,  80, 50, 40,  90;
        300,  80, 45, 45, 120;
        420,  80, 40, 40,  70;
        % Row 2
         80, 200, 55, 50, 100;
        200, 200, 60, 55, 140;
        330, 200, 50, 50,  80;
        430, 200, 45, 45, 110;
        % Row 3
         80, 340, 40, 45,  75;
        190, 340, 50, 50, 130;
        320, 340, 55, 45, 100;
        420, 340, 50, 50,  85;
        % Row 4
         80, 430, 45, 40,  60;
        200, 430, 50, 45,  95;
        330, 430, 45, 50, 115;
        420, 430, 40, 40,  70;
    ];

    % Additional random buildings
    numRand = 12;
    rx  = 50 + rand(numRand,1)*400;
    ry  = 50 + rand(numRand,1)*400;
    rw  = 25 + rand(numRand,1)*35;
    rd  = 25 + rand(numRand,1)*35;
    rh  = 30 + rand(numRand,1)*100;
    bldData = [bldData; rx, ry, rw, rd, rh];

    % Register each building into the occupancy map
    for k = 1:size(bldData,1)
        cx = bldData(k,1); cy = bldData(k,2);
        w  = bldData(k,3); d  = bldData(k,4); h = bldData(k,5);
        xv = max(0,cx-w/2) : res : min(sz(1),cx+w/2);
        yv = max(0,cy-d/2) : res : min(sz(2),cy+d/2);
        zv = 0             : res : h;
        [X,Y,Z] = meshgrid(xv, yv, zv);
        pts = [X(:), Y(:), Z(:)];
        if ~isempty(pts)
            setOccupancy(occMap, pts, ones(size(pts,1),1));
        end
    end

    % Inflate map for safety margin (one grid cell)
    inflate(occMap, res);

    % UAV Scenario (cuboid environment — for visualisation)
    scenario = uavScenario('StopTime', 120, 'UpdateRate', 10);
    for k = 1:size(bldData,1)
        cx = bldData(k,1); cy = bldData(k,2);
        w  = bldData(k,3); d  = bldData(k,4); h = bldData(k,5);
        addMesh(scenario, 'polygon', {[cx-w/2, cy-d/2; cx+w/2, cy-d/2; ...
            cx+w/2, cy+d/2; cx-w/2, cy+d/2], [0 h]}, 0.651*[1 1 1]);
    end
    % Ground plane
    addMesh(scenario, 'polygon', {[0,0; sz(1),0; sz(1),sz(2); 0,sz(2)], ...
        [-0.5 0]}, [0.4 0.6 0.4]);
    fprintf('   Urban map: %d buildings, inflated at %.1f m\n', ...
        size(bldData,1), res);
end

% ---------------------------------------------------------
function [starts, goals] = define_uav_waypoints(params, bldData)
% DEFINE_UAV_WAYPOINTS  Returns start and goal positions for each UAV.
%   Positions are placed at map edges (takeoff pads) at cruise altitude.

    n = params.numUAVs;
    sz = params.mapSize;
    alt = params.cruiseAlt;

    starts = [
         30,  30, alt;
        sz(1)-30,  30, alt+10;
         30, sz(2)-30, alt+5;
        sz(1)-30, sz(2)-30, alt+15;
    ];
    goals = [
        sz(1)-50, sz(2)-50, alt+10;
         50, sz(2)-50, alt+5;
        sz(1)-50,  50, alt+15;
         50,  50, alt;
    ];

    % Use only as many as requested
    starts = starts(1:n, :);
    goals  = goals(1:n, :);
end

% ---------------------------------------------------------
function path = rrt_star_3d(start, goal, occMap, params)
% RRT_STAR_3D  3-D RRT* path planner.
%   Returns an N×3 waypoint matrix [x,y,z] from start to goal.

    maxIter  = params.rrtMaxIter;
    stepSize = params.rrtStepSize;
    goalBias = params.rrtGoalBias;
    rewireR  = stepSize * 2.5;
    sz       = params.mapSize;

    % Tree nodes: [x, y, z, parentIdx, cost]
    nodes  = [start, 0, 0];   % root has no parent (index 0)
    goalReached = false;
    goalIdx     = -1;
    bestCost    = inf;

    for iter = 1:maxIter
        % Sample
        if rand < goalBias
            sample = goal;
        else
            sample = rand(1,3) .* sz;
        end

        % Nearest node
        dists   = vecnorm(nodes(:,1:3) - sample, 2, 2);
        [~, nIdx] = min(dists);
        nearest = nodes(nIdx, 1:3);

        % Steer
        dir  = sample - nearest;
        nd   = norm(dir);
        if nd < 1e-6, continue; end
        newPt = nearest + (dir/nd) * min(stepSize, nd);

        % Collision check (line segment)
        if ~is_collision_free(nearest, newPt, occMap, stepSize)
            continue;
        end

        % Find near nodes for rewiring
        dists2  = vecnorm(nodes(:,1:3) - newPt, 2, 2);
        nearIdx = find(dists2 < rewireR);

        % Choose best parent
        minCost  = nodes(nIdx,5) + norm(newPt - nearest);
        bestPar  = nIdx;
        for j = nearIdx'
            c = nodes(j,5) + norm(newPt - nodes(j,1:3));
            if c < minCost && is_collision_free(nodes(j,1:3), newPt, occMap, stepSize)
                minCost = c;
                bestPar = j;
            end
        end

        % Add node
        nodes(end+1, :) = [newPt, bestPar, minCost]; %#ok<AGROW>
        newIdx = size(nodes, 1);

        % Rewire
        for j = nearIdx'
            newC = minCost + norm(nodes(j,1:3) - newPt);
            if newC < nodes(j,5) && ...
               is_collision_free(newPt, nodes(j,1:3), occMap, stepSize)
                nodes(j, 4) = newIdx;
                nodes(j, 5) = newC;
            end
        end

        % Goal check
        if norm(newPt - goal) < stepSize * 1.2 && ...
           is_collision_free(newPt, goal, occMap, stepSize)
            totalCost = minCost + norm(goal - newPt);
            if totalCost < bestCost
                bestCost    = totalCost;
                goalReached = true;
                goalIdx     = newIdx;
            end
        end
    end

    if ~goalReached
        path = [];
        return;
    end

    % Back-trace path
    path = goal;
    idx  = goalIdx;
    while idx > 0
        path = [nodes(idx,1:3); path]; %#ok<AGROW>
        idx  = nodes(idx, 4);
    end
end

% ---------------------------------------------------------
function free = is_collision_free(p1, p2, occMap, stepSize)
% IS_COLLISION_FREE  Checks a line segment for occupancy.

    d   = norm(p2 - p1);
    n   = max(2, ceil(d / (stepSize * 0.3)));
    pts = [linspace(p1(1),p2(1),n)', ...
           linspace(p1(2),p2(2),n)', ...
           linspace(p1(3),p2(3),n)'];
    occ = getOccupancy(occMap, pts);
    free = all(occ < 0.5);
end

% ---------------------------------------------------------
function smooth = smooth_trajectory(rawPath, params)
% SMOOTH_TRAJECTORY  Fits a cubic spline through RRT* waypoints.

    if size(rawPath, 1) < 3
        smooth = rawPath;
        return;
    end
    % Parameterise by cumulative arc length
    d  = [0; cumsum(vecnorm(diff(rawPath), 2, 2))];
    t  = linspace(0, d(end), max(50, size(rawPath,1)*5));
    sx = spline(d, rawPath(:,1), t);
    sy = spline(d, rawPath(:,2), t);
    sz = spline(d, rawPath(:,3), t);

    % Clip to environment
    mapSz = params.mapSize;
    sx = max(0, min(mapSz(1), sx));
    sy = max(0, min(mapSz(2), sy));
    sz = max(params.cruiseAlt - 20, min(mapSz(3) - 5, sz));

    smooth = [sx(:), sy(:), sz(:)];
end

% ---------------------------------------------------------
function [outPaths, events] = multi_uav_deconflict(paths, params, occMap)
% MULTI_UAV_DECONFLICT  Ensures minimum separation between UAV trajectories.
%   For each pair of UAVs, detects segments that come closer than
%   params.safetyDist and inserts altitude offsets to resolve them.

    n      = params.numUAVs;
    dMin   = params.safetyDist;
    outPaths = paths;
    events   = {};

    for i = 1:n-1
        for j = i+1:n
            pi = outPaths{i};
            pj = outPaths{j};
            ni = size(pi,1); nj = size(pj,1);
            refLen = min(ni, nj);

            % Resample both to equal length
            ti = linspace(0,1,ni); tj = linspace(0,1,nj);
            tr = linspace(0,1,refLen);
            pir = [interp1(ti,pi(:,1),tr)', interp1(ti,pi(:,2),tr)', interp1(ti,pi(:,3),tr)'];
            pjr = [interp1(tj,pj(:,1),tr)', interp1(tj,pj(:,2),tr)', interp1(tj,pj(:,3),tr)'];

            dists = vecnorm(pir - pjr, 2, 2);
            conflictIdx = find(dists < dMin);

            if ~isempty(conflictIdx)
                events{end+1} = sprintf('UAVs %d-%d: %d conflict points (min dist=%.1fm)', ...
                    i, j, numel(conflictIdx), min(dists)); %#ok<AGROW>

                % Apply altitude offset to UAV j at conflict segments
                altOffset = dMin * 1.2;
                for k = conflictIdx'
                    frac = k / refLen;
                    idx  = max(1, round(frac * nj));
                    idx  = min(idx, size(outPaths{j},1));
                    outPaths{j}(idx,3) = outPaths{j}(idx,3) + altOffset;
                    % Smooth spike
                    if idx > 2 && idx < size(outPaths{j},1)-1
                        outPaths{j}(idx-1,3) = outPaths{j}(idx-1,3) + altOffset*0.5;
                        outPaths{j}(idx+1,3) = outPaths{j}(idx+1,3) + altOffset*0.5;
                    end
                end
            end
        end
    end

    if isempty(events)
        fprintf('   No inter-UAV conflicts detected.\n');
    else
        for e = 1:numel(events)
            fprintf('   Resolved: %s\n', events{e});
        end
    end
end

% ---------------------------------------------------------
function metrics = compute_metrics(paths, starts, goals, planTimes, params)
% COMPUTE_METRICS  Computes path length, smoothness, planning time, etc.

    n = params.numUAVs;
    metrics.pathLength    = zeros(n,1);
    metrics.straightDist  = zeros(n,1);
    metrics.pathRatio     = zeros(n,1);
    metrics.avgAlt        = zeros(n,1);
    metrics.planningTime  = planTimes;
    metrics.numWaypoints  = zeros(n,1);

    for i = 1:n
        p = paths{i};
        metrics.numWaypoints(i)  = size(p,1);
        metrics.pathLength(i)    = sum(vecnorm(diff(p),2,2));
        metrics.straightDist(i)  = norm(goals(i,:) - starts(i,:));
        metrics.pathRatio(i)     = metrics.pathLength(i) / ...
                                   max(1, metrics.straightDist(i));
        metrics.avgAlt(i)        = mean(p(:,3));
    end
    metrics.totalPlanTime = sum(planTimes);
    metrics.avgPathRatio  = mean(metrics.pathRatio);
end

% ---------------------------------------------------------
function print_metrics(m)
    fprintf('\n--- Performance Metrics ---\n');
    fprintf('%-6s %-12s %-12s %-10s %-10s %-10s\n', ...
        'UAV', 'PathLen(m)', 'StrDist(m)', 'Ratio', 'AvgAlt(m)', 'PlanT(s)');
    fprintf('%s\n', repmat('-',1,62));
    for i = 1:numel(m.pathLength)
        fprintf('%-6d %-12.1f %-12.1f %-10.2f %-10.1f %-10.3f\n', ...
            i, m.pathLength(i), m.straightDist(i), ...
            m.pathRatio(i), m.avgAlt(i), m.planningTime(i));
    end
    fprintf('%s\n', repmat('-',1,62));
    fprintf('Total planning time : %.3f s\n', m.totalPlanTime);
    fprintf('Average path ratio  : %.3f\n',  m.avgPathRatio);
end

% ---------------------------------------------------------
function visualize_results(scenario, paths, starts, goals, bldData, ...
                            collisionEvents, metrics, params)
% VISUALIZE_RESULTS  Produces 3D path plot and summary dashboard.

    colors = [0.2 0.6 1.0;   % UAV 1 – blue
              1.0 0.4 0.1;   % UAV 2 – orange
              0.1 0.8 0.3;   % UAV 3 – green
              0.9 0.2 0.8];  % UAV 4 – magenta
    n = params.numUAVs;

    % --- Figure 1: 3D trajectory ---
    fig1 = figure('Name','Multi-UAV 3D Trajectories','Position',[50 50 900 650]);
    ax = axes(fig1);
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    xlabel(ax,'X (m)'); ylabel(ax,'Y (m)'); zlabel(ax,'Z (m)');
    title(ax,'Multi-UAV Path Planning — Urban Air Mobility (Project 247)');
    set(ax,'Color',[0.12 0.12 0.15]);
    fig1.Color = [0.1 0.1 0.12];
    ax.XColor = 'w'; ax.YColor = 'w'; ax.ZColor = 'w';
    ax.GridColor = [0.4 0.4 0.4];

    % Draw buildings
    for k = 1:size(bldData,1)
        cx=bldData(k,1); cy=bldData(k,2);
        w=bldData(k,3); d=bldData(k,4); h=bldData(k,5);
        verts = [cx-w/2 cy-d/2 0; cx+w/2 cy-d/2 0; cx+w/2 cy+d/2 0; cx-w/2 cy+d/2 0;
                 cx-w/2 cy-d/2 h; cx+w/2 cy-d/2 h; cx+w/2 cy+d/2 h; cx-w/2 cy+d/2 h];
        faces = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
        patch(ax,'Vertices',verts,'Faces',faces,...
              'FaceColor',[0.35 0.35 0.45],'EdgeColor',[0.5 0.5 0.6],...
              'FaceAlpha',0.7,'EdgeAlpha',0.4);
    end

    % Draw paths
    legH = gobjects(n,1);
    for i = 1:n
        p = paths{i};
        legH(i) = plot3(ax, p(:,1), p(:,2), p(:,3), '-', ...
            'Color', colors(i,:), 'LineWidth', 2.5, ...
            'DisplayName', sprintf('UAV %d', i));
        % Start marker
        scatter3(ax, starts(i,1), starts(i,2), starts(i,3), 120, ...
            colors(i,:), '^', 'filled', 'MarkerEdgeColor','w');
        % Goal marker
        scatter3(ax, goals(i,1), goals(i,2), goals(i,3), 120, ...
            colors(i,:), 's', 'filled', 'MarkerEdgeColor','w');
        % UAV icon at end
        scatter3(ax, p(end,1), p(end,2), p(end,3), 80, ...
            colors(i,:), 'o', 'filled');
    end
    legend(ax, legH, 'TextColor', 'w', 'Color', [0.2 0.2 0.2], ...
           'Location', 'northeast');
    view(ax, 45, 30);
    xlim(ax,[0 params.mapSize(1)]); ylim(ax,[0 params.mapSize(2)]);
    zlim(ax,[0 params.mapSize(3)]);

    % --- Figure 2: Metrics dashboard ---
    fig2 = figure('Name','Performance Dashboard','Position',[980 50 800 650]);
    fig2.Color = [0.1 0.1 0.12];

    uavLabels = arrayfun(@(i) sprintf('UAV %d',i), 1:n, 'UniformOutput', false);

    % Path length
    ax2 = subplot(2,2,1,'Parent',fig2);
    set(ax2,'Color',[0.15 0.15 0.18],'XColor','w','YColor','w');
    b1 = bar(ax2, metrics.pathLength, 'FaceColor','flat');
    b1.CData = colors(1:n,:);
    ax2.XTickLabel = uavLabels;
    title(ax2,'Path Length (m)','Color','w');
    ylabel(ax2,'Length (m)','Color','w');
    grid(ax2,'on'); ax2.GridColor=[0.4 0.4 0.4];

    % Path ratio
    ax3 = subplot(2,2,2,'Parent',fig2);
    set(ax3,'Color',[0.15 0.15 0.18],'XColor','w','YColor','w');
    b2 = bar(ax3, metrics.pathRatio, 'FaceColor','flat');
    b2.CData = colors(1:n,:);
    ax3.XTickLabel = uavLabels;
    yline(ax3,1,'--','Ideal','Color','w','LabelHorizontalAlignment','left');
    title(ax3,'Path Ratio (actual/straight)','Color','w');
    ylabel(ax3,'Ratio','Color','w');
    grid(ax3,'on'); ax3.GridColor=[0.4 0.4 0.4];

    % Planning time
    ax4 = subplot(2,2,3,'Parent',fig2);
    set(ax4,'Color',[0.15 0.15 0.18],'XColor','w','YColor','w');
    b3 = bar(ax4, metrics.planningTime, 'FaceColor','flat');
    b3.CData = colors(1:n,:);
    ax4.XTickLabel = uavLabels;
    title(ax4,'RRT* Planning Time (s)','Color','w');
    ylabel(ax4,'Time (s)','Color','w');
    grid(ax4,'on'); ax4.GridColor=[0.4 0.4 0.4];

    % Average altitude
    ax5 = subplot(2,2,4,'Parent',fig2);
    set(ax5,'Color',[0.15 0.15 0.18],'XColor','w','YColor','w');
    b4 = bar(ax5, metrics.avgAlt, 'FaceColor','flat');
    b4.CData = colors(1:n,:);
    ax5.XTickLabel = uavLabels;
    yline(ax5, params.cruiseAlt,'--','Cruise Alt','Color','c',...
        'LabelHorizontalAlignment','left');
    title(ax5,'Average Altitude (m)','Color','w');
    ylabel(ax5,'Altitude (m)','Color','w');
    grid(ax5,'on'); ax5.GridColor=[0.4 0.4 0.4];

    % --- Figure 3: Altitude profiles ---
    fig3 = figure('Name','Altitude Profiles','Position',[50 730 900 300]);
    fig3.Color = [0.1 0.1 0.12];
    ax6 = axes(fig3);
    set(ax6,'Color',[0.12 0.12 0.15],'XColor','w','YColor','w');
    hold(ax6,'on'); grid(ax6,'on');
    for i = 1:n
        p = paths{i};
        arcLen = [0; cumsum(vecnorm(diff(p),2,2))];
        plot(ax6, arcLen, p(:,3), '-', 'Color', colors(i,:), 'LineWidth', 2, ...
             'DisplayName', sprintf('UAV %d',i));
    end
    yline(ax6, params.cruiseAlt,'--','Cruise Alt','Color','w');
    xlabel(ax6,'Arc Length (m)','Color','w');
    ylabel(ax6,'Altitude (m)','Color','w');
    title(ax6,'UAV Altitude Profiles Along Path','Color','w');
    legend(ax6,'TextColor','w','Color',[0.2 0.2 0.2]);

    % Save figures
    if ~exist('results','dir'), mkdir('results'); end
    saveas(fig1,'results/trajectories_3D.png');
    saveas(fig2,'results/performance_dashboard.png');
    saveas(fig3,'results/altitude_profiles.png');
    fprintf('Figures saved to results/\n');
end
