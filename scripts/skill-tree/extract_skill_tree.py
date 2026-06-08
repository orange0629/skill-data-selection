import openai
import json
import jsonlines
from collections import defaultdict

# Configuration
# Set OpenAI API key
client = openai.OpenAI(
    api_key="",
)

INPUT_FILE = "MATH/train.jsonl"
OUTPUT_JSON = "skill_tree.json"
HTML_TEMPLATE = "visualization.html"

class SkillTreeGenerator:
    def __init__(self):
        self.tree = {"name": "Root", "children": defaultdict(dict), "parent": None, "descendants": defaultdict(dict), "meta_data_list": []}
        self.node_registry = defaultdict(dict)
        self.all_nodes = defaultdict(list)

    def _call_gpt(self, prompt: str, temp=0.3) -> str:
        """Call OpenAI API with temperature control"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        return response.choices[0].message.content.strip()

    def decompose_solution(self, solution: str) -> list:
        """Extract steps from solution text"""
        prompt = f"""Decompose this solution into atomic steps as numbered list:
        
{solution}

Output format:
1. [Step 1 description]
2. [Step 2 description]
..."""
        
        response = self._call_gpt(prompt, temp=0.4)
        return [line.split('. ', 1)[-1] for line in response.split('\n') if line.strip() and ". " in line]

    def generate_skill_chain(self, step: str) -> list:
        """Generate hierarchical skill chain for a step"""
        prompt = f"""Generate abstract-to-concrete skill chain for this step (max 8 levels):

Examples:
Step: Calculate discriminant D=b²-4ac
Chain: Mathematics > Algebra > Equation Solving > Quadratic Equations > Discriminant Calculation

Current step: {step}
Output format: Category levels separated by >"""
        
        chain_str = self._call_gpt(prompt)
        return chain_str.split(' > ')

    def update_chain_to_tree(self, chain: list, meta_data):
        """Build hierarchical tree from skill chains ensuring unique nodes"""
        chain = ["Root"] + chain
        current = self.tree
        path_so_far = []

        for level in chain:
            path_so_far.append(level)
            path_key = " > ".join(path_so_far)

            # If node already exists in registry, reuse it
            # Change to vague match later
            if level == current["name"]:
                current["meta_data_list"].append(meta_data)
            elif level in current["descendants"]:
                tmp_source = current
                current = current["descendants"][level]
                current["meta_data_list"].append(meta_data)
                tmp_current_parent = current["parent"]
                while tmp_current_parent is not None and tmp_current_parent["name"] != tmp_source["name"]:
                    tmp_current_parent["meta_data_list"].append(meta_data)
                    tmp_current_parent = tmp_current_parent["parent"]
            else:
                # Create new node and register it
                new_node = {
                    "name": level,
                    "children": defaultdict(dict),
                    "parent": current,
                    "descendants": defaultdict(dict),
                    "meta_data_list": [meta_data]
                }
                current["children"][level] = new_node
                current = new_node
                tmp_current_parent = current["parent"]
                while tmp_current_parent is not None:
                    tmp_current_parent["descendants"][level] = new_node
                    tmp_current_parent = tmp_current_parent["parent"]
                
                if level in self.all_nodes:
                    for potential_candidate in self.all_nodes[level]:
                        prev_parent = potential_candidate["parent"]
                        tmp_parent = current["parent"]
                        while tmp_parent != None and tmp_parent["name"] != prev_parent["name"]:
                            tmp_parent = tmp_parent["parent"]
                        if tmp_parent != None and tmp_parent["name"] == prev_parent["name"]:
                            new_node["children"].update(potential_candidate["children"])
                            new_node["descendants"].update(potential_candidate["descendants"])
                            new_node["meta_data_list"] += potential_candidate["meta_data_list"]
                            try:
                                del prev_parent["children"][level] # Potential bug here. Need fixing sometime.
                            except:
                                pass
                            break
                self.all_nodes[level].append(new_node)

    def _convert_structure(self, node: dict) -> dict:
        """Convert nested defaultdict to standard dict"""
        return {
            "name": node["name"],
            "children": [self._convert_structure(child) for child in node["children"].values()],
            "relevant_instance": [meta_data["instance"]['unique_id'] for meta_data in node["meta_data_list"]]
        }

    def process_jsonl(self, file_path: str):
        """Process JSONL file and build skill tree"""
        all_chains = []
        count = 0
        
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                if count >= 3:
                    break
                # Extract steps from solution
                steps = self.decompose_solution(obj['solution'])
                print(f"Processing: {obj['unique_id']}")
                print(f"Extracted steps: {steps}")

                # Generate skill chains
                for step in steps:
                    chain = self.generate_skill_chain(step)
                    self.update_chain_to_tree(chain, {"step": step, "instance": obj})
                    all_chains.append(chain)
                    print(f"Step: {step[:100]}... => Chain: {' > '.join(chain)}")
                count += 1

        # Build final tree
        return self._convert_structure(self.tree), self.tree

# Visualization template
HTML_TEMPLATE_CONTENT = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {
            font-family: "Segoe UI", sans-serif;
            margin: 0;
        }
        .node circle {
            fill: #fff;
            stroke: steelblue;
            stroke-width: 2px;
        }
        .node text {
            font-size: 14px;
        }
        .link {
            fill: none;
            stroke: #ccc;
            stroke-width: 1.5px;
        }
        svg {
            width: 100vw;
            height: 100vh;
        }
        .tooltip {
            position: absolute;
            text-align: center;
            padding: 6px;
            font-size: 13px;
            background: rgba(0, 0, 0, 0.75);
            color: #fff;
            border-radius: 4px;
            pointer-events: none;
            visibility: hidden;
        }
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <svg></svg>
    <div class="tooltip" id="tooltip"></div>

    <script>
        const svg = d3.select("svg"),
              g = svg.append("g").attr("transform", "translate(100,100)");

        const tooltip = d3.select("#tooltip");

        const zoom = d3.zoom()
            .scaleExtent([0.2, 3])
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
            });

        svg.call(zoom);

        d3.json("%s").then(data => {
            const root = d3.hierarchy(data);
            const treeLayout = d3.tree().nodeSize([40, 180]);
            treeLayout(root);

            // Draw links
            g.selectAll(".link")
                .data(root.links())
                .enter().append("path")
                .attr("class", "link")
                .attr("d", d3.linkHorizontal()
                    .x(d => d.y)
                    .y(d => d.x));

            // Draw nodes
            const nodes = g.selectAll(".node")
                .data(root.descendants())
                .enter().append("g")
                .attr("class", "node")
                .attr("transform", d => `translate(${d.y},${d.x})`);

            nodes.append("circle")
                .attr("r", 8)
                .attr("fill", d => d3.interpolateRainbow(d.depth / 6));

            nodes.append("text")
                .attr("dy", 0)
                .attr("x", d => d.children ? -15 : 15)
                .attr("text-anchor", d => d.children ? "end" : "start")
                .style("font-size", "14px")
                .text(d => d.data.name)
                .on("mouseover", function(event, d) {
                    tooltip.style("visibility", "visible")
                           .text(d.data.name);
                })
                .on("mousemove", function(event) {
                    tooltip.style("top", (event.pageY + 10) + "px")
                           .style("left", (event.pageX + 10) + "px");
                })
                .on("mouseout", function() {
                    tooltip.style("visibility", "hidden");
                })
                .call(wrap, 180);
        });

        // Helper for text wrapping
        function wrap(texts, width) {
            texts.each(function() {
                const text = d3.select(this);
                const words = text.text().split(/\\s+/).reverse();
                let word;
                let line = [];
                let lineNumber = 0;
                const x = text.attr("x");
                const dy = 1.1;
                let tspan = text.text(null)
                    .append("tspan")
                    .attr("x", x)
                    .attr("dy", "0em");

                while (word = words.pop()) {
                    line.push(word);
                    tspan.text(line.join(" "));
                    if (tspan.node().getComputedTextLength() > width) {
                        line.pop();
                        tspan.text(line.join(" "));
                        line = [word];
                        tspan = text.append("tspan")
                            .attr("x", x)
                            .attr("dy", ++lineNumber * dy + "em")
                            .text(word);
                    }
                }
            });
        }
    </script>
</body>
</html>
""" % OUTPUT_JSON

if __name__ == "__main__":
    # Generate skill tree
    generator = SkillTreeGenerator()
    final_tree, final_tree_raw = generator.process_jsonl(INPUT_FILE)

    # Save JSON output
    # with open("raw_tree.json", "w") as f:
    #     json.dump(final_tree_raw, f, indent=2)
    
    # Save JSON output
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_tree, f, indent=2)
    
    # Create visualization file
    with open(HTML_TEMPLATE, "w") as f:
        f.write(HTML_TEMPLATE_CONTENT)
    
    print(f"Skill tree saved to {OUTPUT_JSON}")
    print(f"Open {HTML_TEMPLATE} to view interactive visualization")