/**
 * AIGText — Knowledge Graph (v3)
 * Radial force-directed: center root, orbiting categories/subclasses,
 * repulsion between nodes, drag, scroll-wheel zoom.
 */
(function () {
  "use strict";

  // ── DOM ──
  var dom = {
    canvas:       document.getElementById("graph-canvas"),
    backBtn:      document.getElementById("graph-back-btn"),
    searchBtn:    document.getElementById("graph-search-btn"),
    searchBar:    document.getElementById("graph-search-bar"),
    searchInput:  document.getElementById("graph-search-input"),
    searchClear:  document.getElementById("graph-search-clear"),
    searchResults:document.getElementById("graph-search-results"),
    resetBtn:     document.getElementById("graph-reset-btn"),
    stats:        document.getElementById("graph-stats"),
    loading:      document.getElementById("graph-loading"),
    detailPanel:  document.getElementById("graph-detail-panel"),
    detailTitle:  document.getElementById("graph-detail-title"),
    detailLevel:  document.getElementById("graph-detail-level"),
    detailDesc:   document.getElementById("graph-detail-desc"),
    detailRel:    document.getElementById("graph-detail-relations"),
    detailClose:  document.getElementById("graph-detail-close"),
  };

  var S = { root: null, svg: null, g: null, sim: null, zoom: null, nodes: [], links: [], activeId: null, nodeMap: {}, cx: 0, cy: 0 };

  // ── Layout constants ──
  var ELLIPSE = { root: [110, 28], coarse: [85, 24], sub: [68, 20] };
  var R1 = 230; // coarse ring
  var R2 = 420; // subclass ring

  // ── Init ──
  dom.backBtn.addEventListener("click", function () { Nav.go("index.html"); });
  dom.searchBtn.addEventListener("click", toggleSearch);
  dom.searchClear.addEventListener("click", function () { dom.searchInput.value = ""; onSearch(); });
  dom.searchInput.addEventListener("input", onSearch);
  dom.resetBtn.addEventListener("click", resetView);
  dom.detailClose.addEventListener("click", closeDetail);
  dom.canvas.addEventListener("click", function (e) { if (e.target === dom.canvas) closeDetail(); });
  window.addEventListener("resize", onResize);
  fetchData();

  // Transform API data: flat taxonomy array → hierarchical tree
  // taxonomy.json is an array of { doc_id, coarse_category, sub_category, document_path }
  function normalizeData(raw) {
    var root = {
      id: "root",
      name: "物联网设备知识图谱",
      desc: "",
      level: "root",
      children: []
    };

    // Group entries by coarse_category, preserving first-seen order
    var coarseMap = {};
    var coarseOrder = [];
    var arr = Array.isArray(raw) ? raw : [];
    arr.forEach(function (entry) {
      var cc = entry.coarse_category || "未分类";
      if (!coarseMap[cc]) {
        coarseMap[cc] = { seen: false, subs: [] };
      }
      if (!coarseMap[cc].seen) {
        coarseMap[cc].seen = true;
        coarseOrder.push(cc);
      }
      // Collect unique sub_category under each coarse
      var sc = entry.sub_category || cc;
      if (coarseMap[cc].subs.indexOf(sc) === -1) {
        coarseMap[cc].subs.push(sc);
      }
    });

    coarseOrder.forEach(function (cc) {
      var cn = {
        id: cc,
        name: cc,
        desc: "",
        level: "coarse",
        children: []
      };
      coarseMap[cc].subs.forEach(function (sc) {
        cn.children.push({
          id: sc,
          name: sc,
          desc: "",
          level: "sub",
          children: []
        });
      });
      root.children.push(cn);
    });

    return root;
  }

  function fetchData() {
    fetch("../data/iot_knowledge/taxonomy.json")
      .then(function (r) { if (!r.ok) throw Error(r.status); return r.json(); })
      .then(function (data) {
        S.root = normalizeData(data);
        build();
      })
      .catch(function (e) { dom.loading.innerHTML = '<p style="color:#f85149">' + e.message + '</p>'; });
  }

  // ── Build ──
  function build() {
    // Safety checks for data structure
    if (!S.root || !S.root.children) {
      console.error('Invalid graph data:', S.root);
      dom.loading.innerHTML = '<p style="color:#f85149">数据格式错误</p>';
      return;
    }

    var cc = S.root.children.length;
    var sc = 0;
    S.root.children.forEach(function (c) { sc += (c.children || []).length; });
    dom.stats.textContent = cc + " 设备大类 · " + sc + " 具体型号";

    var W = dom.canvas.clientWidth;
    var H = dom.canvas.clientHeight;
    S.cx = W / 2; S.cy = H / 2;

    // SVG
    var svg = d3.select(dom.canvas).append("svg").attr("width", W).attr("height", H);

    // Defs: gradients + grid
    var defs = svg.append("defs");

    // Node gradients
    var g1 = defs.append("radialGradient").attr("id", "rootGrad");
    g1.append("stop").attr("offset", "0%").attr("stop-color", "#dbeafe");
    g1.append("stop").attr("offset", "100%").attr("stop-color", "#eff6ff");
    var g2 = defs.append("radialGradient").attr("id", "coarseGrad");
    g2.append("stop").attr("offset", "0%").attr("stop-color", "#ede9fe");
    g2.append("stop").attr("offset", "100%").attr("stop-color", "#f5f3ff");
    var g3 = defs.append("radialGradient").attr("id", "subGrad");
    g3.append("stop").attr("offset", "0%").attr("stop-color", "#fff");
    g3.append("stop").attr("offset", "100%").attr("stop-color", "#f9fafb");

    // Grid pattern
    defs.append("pattern").attr("id", "g").attr("width", 40).attr("height", 40).attr("patternUnits", "userSpaceOnUse")
      .append("circle").attr("cx", 20).attr("cy", 20).attr("r", 0.5).attr("fill", "rgba(0,0,0,0.05)");
    svg.append("rect").attr("width", "100%").attr("height", "100%").attr("fill", "url(#g)");

    // Pulsing rings around center
    svg.append("circle").attr("class", "pulse-ring pulse-ring--outer").attr("cx", S.cx).attr("cy", S.cy);
    svg.append("circle").attr("class", "pulse-ring pulse-ring--inner").attr("cx", S.cx).attr("cy", S.cy);

    // Ring guides
    [R1, R2].forEach(function (r) {
      svg.append("circle").attr("cx", S.cx).attr("cy", S.cy).attr("r", r)
        .attr("fill", "none").attr("stroke", "rgba(139,92,246,0.1)").attr("stroke-width", 1).attr("stroke-dasharray", "6 12");
    });

    var g = svg.append("g");
    S.svg = svg;
    S.g = g;
    S.nodeMap = {};

    // ── Build nodes + links ──
    var nodes = [], links = [];

    var rootId = S.root.name;
    nodes.push({
      id: rootId, name: S.root.name, desc: S.root.desc || "", level: "root",
      parentId: null, children: [], x: S.cx, y: S.cy
    });
    S.nodeMap[rootId] = nodes[0];

    S.root.children.forEach(function (cat, i) {
      var angle = (i / S.root.children.length) * 2 * Math.PI - Math.PI / 2;
      var cn = {
        id: cat.id, name: cat.name, desc: cat.desc || "", level: "coarse",
        parentId: rootId, children: [],
        x: S.cx + R1 * Math.cos(angle), y: S.cy + R1 * Math.sin(angle)
      };
      nodes.push(cn);
      S.nodeMap[cat.id] = cn;
      links.push({ source: rootId, target: cat.id });

      (cat.children || []).forEach(function (sub, j) {
        var sa = angle + (j - (cat.children.length - 1) / 2) * 0.22;
        var sn = {
          id: sub.id, name: sub.name, desc: sub.desc || "", level: "sub",
          parentId: cat.id, children: [],
          x: S.cx + R2 * Math.cos(sa), y: S.cy + R2 * Math.sin(sa)
        };
        nodes.push(sn);
        S.nodeMap[sub.id] = sn;
        cn.children.push(sub.id);
        links.push({ source: cat.id, target: sub.id });
      });
    });

    S.nodes = nodes;
    S.links = links;

    // ── Links ──
    var linkG = g.append("g");
    var linkEls = linkG.selectAll("line").data(links).join("line")
      .attr("class", "graph-link")
      .attr("stroke", function (l) {
        var sid = typeof l.source === "object" ? l.source.id : l.source;
        var sn = S.nodeMap[sid];
        return sn && sn.level === "root" ? "#c4b5fd" : "#ddd6fe";
      })
      .attr("stroke-width", function (l) {
        var sid = typeof l.source === "object" ? l.source.id : l.source;
        return S.nodeMap[sid] && S.nodeMap[sid].level === "root" ? 1.5 : 0.8;
      });

    // ── Nodes ──
    var nodeG = g.append("g");
    var nodeEls = nodeG.selectAll("g").data(nodes).join("g")
      .attr("class", function (n) { return "graph-node graph-node--" + n.level; })
      .call(dragBehavior());

    nodeEls.append("ellipse")
      .attr("class", function (n) { return "graph-node-ellipse graph-node-ellipse--" + n.level; })
      .attr("rx", function (n) { return ELLIPSE[n.level][0]; })
      .attr("ry", function (n) { return ELLIPSE[n.level][1]; })
      .on("click", function (ev, n) { ev.stopPropagation(); selectNode(n); });

    nodeEls.append("text")
      .attr("class", function (n) { return "graph-node-text graph-node-text--" + n.level; })
      .text(function (n) { return n.name; });

    // ── Simulation ──
    var SPRING_STRENGTH = 0.15;
    var sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(function (n) { return n.id; })
        .distance(function (l) {
          var sid = typeof l.source === "object" ? l.source.id : l.source;
          return S.nodeMap[sid] && S.nodeMap[sid].level === "root" ? R1 : 100;
        })
        .strength(function (l) {
          var sid = typeof l.source === "object" ? l.source.id : l.source;
          return S.nodeMap[sid] && S.nodeMap[sid].level === "root" ? 0.3 : 0.15;
        })
      )
      .force("charge", d3.forceManyBody()
        .strength(function (n) {
          if (n.level === "root") return -1500;
          if (n.level === "coarse") return -800;
          return -300;
        })
      )
      .force("collide", d3.forceCollide(function (n) {
        var e = ELLIPSE[n.level];
        return Math.max(e[0], e[1]) + 14;
      }).strength(0.8).iterations(2))
      // Center spring: pulls root node toward (S.cx, S.cy) like a spring.
      // Skipped while the user is actively dragging the root (root.fx !== null).
      .force("spring", function spring(alpha) {
        var root = S.nodeMap[rootId];
        if (!root || root.fx != null) return;
        var dx = S.cx - root.x;
        var dy = S.cy - root.y;
        root.vx += dx * alpha * SPRING_STRENGTH;
        root.vy += dy * alpha * SPRING_STRENGTH;
      })
      .alphaDecay(0.015)
      .on("tick", function () {
        linkEls.attr("x1", function (l) { return l.source.x; })
               .attr("y1", function (l) { return l.source.y; })
               .attr("x2", function (l) { return l.target.x; })
               .attr("y2", function (l) { return l.target.y; });
        nodeEls.attr("transform", function (n) { return "translate(" + n.x + "," + n.y + ")"; });
      });

    S.sim = sim;

    // ── Zoom (scroll wheel, no Ctrl needed) ──
    var zoom = d3.zoom()
      .scaleExtent([0.2, 3])
      .on("zoom", function (ev) { g.attr("transform", ev.transform); });
    svg.call(zoom);
    S.zoom = zoom;
    svg.call(zoom.transform, d3.zoomIdentity.scale(0.75));

    dom.loading.classList.add("hidden");
  }

  // ── Drag ──
  function dragBehavior() {
    return d3.drag()
      .on("start", function (ev, n) {
        if (!ev.active) S.sim.alphaTarget(0.5).restart();
        n.fx = n.x; n.fy = n.y;
        dom.canvas.classList.add("dragging");
      })
      .on("drag", function (ev, n) {
        n.fx = ev.x; n.fy = ev.y;
      })
      .on("end", function (ev, n) {
        if (!ev.active) S.sim.alphaTarget(0);
        n.fx = null; n.fy = null; // snap back into force sim
        dom.canvas.classList.remove("dragging");
      });
  }

  // ── Select / Detail ──
  function selectNode(n) {
    S.activeId = n.id;
    showDetail(n);
    dimOthers(n);
  }

  function showDetail(n) {
    dom.detailTitle.textContent = n.name;
    var lvl = n.level === "root" ? "根节点" : n.level === "coarse" ? "设备大类" : "具体型号";
    dom.detailLevel.textContent = lvl;
    dom.detailLevel.setAttribute("data-level", n.level === "root" ? "root" : n.level === "coarse" ? "coarse" : "sub");
    dom.detailDesc.textContent = n.desc || "暂无描述";

    var html = "";
    if (n.level === "root" && S.root.children) {
      html += '<p class="graph-detail-rel__title">' + S.root.children.length + ' 个设备大类</p>';
      S.root.children.forEach(function (c) {
        html += '<div class="graph-detail-rel__item" data-nid="' + c.id + '">' + c.name + '</div>';
      });
    }
    if (n.level === "coarse" && n.children && n.children.length) {
      html += '<p class="graph-detail-rel__title">' + n.children.length + ' 个型号</p>';
      n.children.forEach(function (cid) {
        var s = S.nodeMap[cid];
        if (s) html += '<div class="graph-detail-rel__item" data-nid="' + cid + '">' + s.name + '</div>';
      });
    }
    if (n.level === "sub" && n.parentId) {
      var p = S.nodeMap[n.parentId];
      if (p) html += '<p class="graph-detail-rel__title">所属</p><div class="graph-detail-rel__item" data-nid="' + p.id + '">' + p.name + '</div>';
    }

    dom.detailRel.innerHTML = html;
    dom.detailRel.querySelectorAll("[data-nid]").forEach(function (el) {
      el.addEventListener("click", function (ev) {
        ev.stopPropagation();
        var dn = S.nodeMap[el.dataset.nid];
        if (dn) selectNode(dn);
      });
    });
    dom.detailPanel.classList.remove("hidden");
  }

  function closeDetail() {
    S.activeId = null;
    dom.detailPanel.classList.add("hidden");
    S.svg.selectAll(".graph-node").classed("graph-node--dimmed", false);
  }

  function dimOthers(n) {
    S.svg.selectAll(".graph-node").classed("graph-node--dimmed", function (on) {
      if (on.id === n.id) return false;
      if (n.children && n.children.indexOf(on.id) !== -1) return false;
      if (on.parentId === n.id) return false;
      if (n.parentId && n.parentId === on.id) return false;
      return true;
    });
  }

  // ── Reset ──
  function resetView() {
    S.nodes.forEach(function (n) { if (n.level !== "root") { n.fx = null; n.fy = null; } });
    S.sim.alpha(0.5).restart();
    S.svg.transition().duration(500).call(S.zoom.transform, d3.zoomIdentity.scale(0.75));
    closeDetail();
  }

  // ── Search ──
  function toggleSearch() {
    var show = dom.searchBar.classList.contains("hidden");
    if (show) {
      dom.searchBar.classList.remove("hidden");
      dom.searchBtn.classList.add("graph-action-btn--active");
      setTimeout(function () { dom.searchInput.focus(); }, 100);
    } else {
      dom.searchBar.classList.add("hidden");
      dom.searchBtn.classList.remove("graph-action-btn--active");
      dom.searchInput.value = "";
      dom.searchResults.classList.add("hidden");
      S.svg.selectAll(".graph-node").classed("graph-node--dimmed", false);
    }
  }

  function onSearch() {
    var q = (dom.searchInput.value || "").trim().toLowerCase();
    if (!q) { dom.searchResults.classList.add("hidden"); S.svg.selectAll(".graph-node").classed("graph-node--dimmed", false); return; }

    var matches = [];
    S.nodes.forEach(function (n) {
      if (n.level === "root") return;
      if (n.name.toLowerCase().indexOf(q) !== -1 || n.id.toLowerCase().indexOf(q) !== -1) matches.push(n);
    });

    if (!matches.length) {
      dom.searchResults.classList.remove("hidden");
      dom.searchResults.innerHTML = '<div style="padding:6px 8px;font-size:12px;color:#9ca3af">无匹配</div>';
      return;
    }

    dom.searchResults.classList.remove("hidden");
    dom.searchResults.innerHTML = matches.map(function (n) {
      return '<div class="graph-detail-rel__item" data-nid="' + n.id + '">' + n.name + '</div>';
    }).join("");

    dom.searchResults.querySelectorAll("[data-nid]").forEach(function (el) {
      el.addEventListener("click", function () {
        var dn = S.nodeMap[el.dataset.nid];
        if (dn) { selectNode(dn); dom.searchResults.classList.add("hidden"); }
      });
    });

    var ids = {};
    matches.forEach(function (n) { ids[n.id] = true; });
    S.svg.selectAll(".graph-node").classed("graph-node--dimmed", function (n) {
      return n.level !== "root" && !ids[n.id];
    });
  }

  // ── Resize ──
  function onResize() {
    if (!S.svg) return;
    var W = dom.canvas.clientWidth;
    var H = dom.canvas.clientHeight;
    S.svg.attr("width", W).attr("height", H);
    S.cx = W / 2;
    S.cy = H / 2;
    // Update pulse rings to track new center
    S.svg.selectAll(".pulse-ring").attr("cx", S.cx).attr("cy", S.cy);
    // Do NOT pin root - the spring force will pull it toward the new center
    S.sim.alpha(0.3).restart();
  }

})();
