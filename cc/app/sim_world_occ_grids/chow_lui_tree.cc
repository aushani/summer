#include "app/sim_world_occ_grids/chow_lui_tree.h"

#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>

#include "library/timer/timer.h"

namespace app {
namespace sim_world_occ_grids {

ChowLuiTree::ChowLuiTree() {
}

ChowLuiTree::ChowLuiTree(const JointModel &jm) : resolution_(jm.GetResolution()) {
  boost::optional<const rt::DenseOccGrid&> dog;

  //// Make edges
  //auto edges = ConstructEdges(jm, dog);

  //// Make tree
  //MakeTree(edges, jm);
  ConstructTreePrim(jm, dog);
}

ChowLuiTree::ChowLuiTree(const JointModel &jm, const rt::DenseOccGrid &dog) : resolution_(jm.GetResolution()) {
  //// Make edges
  //auto edges = ConstructEdges(jm, dog);

  //// Make tree
  //MakeTree(edges, jm);
  ConstructTreePrim(jm, dog);
}

void ChowLuiTree::MakeTree(const std::vector<ChowLuiTree::Edge> &e, const JointModel &jm) {
  // Get tree edges
  std::vector<Edge> edges(e);

  double sum_mi = 0;

  while (!edges.empty()) {
    // Find an edge we can process
    auto it = edges.begin();
    bool found = false;
    bool flip = false;
    for ( ; it != edges.end(); it++) {
      if (nodes_.count(it->loc1) > 0) {
        found = true;
        flip = false;
        break;
      }

      if (nodes_.count(it->loc2) > 0) {
        found = true;
        flip = true;
        break;
      }
    }

    // We need to make a root node first
    if (!found) {
      it = edges.begin();

      Node n(it->loc1, jm);

      nodes_.insert({it->loc1, n});
      root_locs_.push_back(n.GetLocation());

      found = true;
      flip = false;
    }

    Edge edge = *it;
    auto loc1 = (!flip) ? edge.loc1 : edge.loc2;
    auto loc2 = (!flip) ? edge.loc2 : edge.loc1;

    // Create Node
    Node n(loc2, loc1, jm);
    nodes_.insert({loc2, n});

    // Add to parent's children
    auto p_n = nodes_.find(loc1);
    BOOST_ASSERT(p_n != nodes_.end());
    p_n->second.AddChild(n);

    // Remove edge that we processed
    // TODO O(n)
    edges.erase(it);

    sum_mi += -edge.weight;

    // TODO what about nodes with no edges?
  }

  printf("Have %ld nodes, %ld roots in tree, %5.3f sum MI\n", nodes_.size(), root_locs_.size(), sum_mi);

}

std::vector<ChowLuiTree::Edge> ChowLuiTree::ConstructEdges(const JointModel &jm, const boost::optional<const rt::DenseOccGrid&> &dog) {
  library::timer::Timer t;
  t.Start();

  // Adapted from boost example

  // Typedef's for convience
  typedef boost::adjacency_list<boost::vecS, boost::vecS,
          boost::undirectedS, boost::no_property,
          boost::property<boost::edge_weight_t, double> > Graph;
  typedef boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;
  typedef std::pair<int, int> BoostEdge;

  // Making graph for boost;
  std::vector<BoostEdge> edges;
  std::vector<double> weights;

  // Mapping from node id (int) to location
  std::map<rt::Location, int> loc_int_mapping;
  std::vector<rt::Location> int_loc_mapping;

  // Get all edges
  int min_ij = -jm.GetNXY() / 2;
  int max_ij = min_ij + jm.GetNXY();

  int min_k = -jm.GetNZ() / 2;
  int max_k = min_k + jm.GetNZ();

  for (int i1=min_ij; i1 < max_ij; i1++) {
    for (int j1=min_ij; j1 < max_ij; j1++) {
      for (int k1=min_k; k1 < max_k; k1++) {
        rt::Location loc1(i1, j1, k1);

        if (dog && !dog->IsKnown(loc1)) {
          continue;
        }

        if (loc_int_mapping.count(loc1) == 0) {
          loc_int_mapping[loc1] = int_loc_mapping.size();
          int_loc_mapping.push_back(loc1);
        }

        int i_loc1 = loc_int_mapping[loc1];

        if (!jm.InRange(loc1)) {
          //printf("Out of range! %d, %d, %d\n",
          //    loc1.i, loc1.j, loc1.k);
          continue;
        }

        for (int i2 = min_ij; i2 < max_ij; i2++) {
          for (int j2 = min_ij; j2 < max_ij; j2++) {
            for (int k2 = min_k; k2 < max_k; k2++) {
              double di = i1 - i2;
              double dj = j1 - j2;
              double dk = k1 - k2;
              double d2 = di*di + dj*dj + dk*dk;
              if (sqrt(d2) > kMaxDistanceBetweenNodes_) {
                continue;
              }

              rt::Location loc2(i2, j2, k2);

              if (loc1 == loc2 || loc2 < loc1) {
                continue;
              }

              if (dog && !dog->IsKnown(loc2)) {
                continue;
              }

              if (jm.GetNumObservations(loc1, loc2) < kMinObservations_) {
                continue;
              }

              if (loc_int_mapping.count(loc2) == 0) {
                loc_int_mapping[loc2] = int_loc_mapping.size();
                int_loc_mapping.push_back(loc2);
              }

              int i_loc2 = loc_int_mapping[loc2];

              double mi = jm.ComputeMutualInformation(loc1, loc2);
              if (mi < kMinMutualInformation_) {
                continue;
              }

              double weight = -mi; // because we have minimum spanning tree but want max

              edges.push_back(BoostEdge(i_loc1, i_loc2));
              weights.push_back(weight);
            }
          }
        }
      }
    }
  }
  size_t num_edges = edges.size();
  size_t num_nodes = int_loc_mapping.size();
  //printf("Took %5.3f seconds to get %ld edges, %ld nodes\n", t.GetSeconds(), num_edges, num_nodes);

  Graph g(edges.begin(), edges.begin() + num_edges, weights.begin(), num_nodes);

  boost::property_map<Graph, boost::edge_weight_t>::type weight = get(boost::edge_weight, g);
  std::vector<EdgeDescriptor> spanning_tree;

  t.Start();
  boost::kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));
  //printf("kruskal done in %5.3f sec, have %ld edges\n", t.GetSeconds(), spanning_tree.size());

  std::vector<Edge> clt_edges;
  for (std::vector < EdgeDescriptor >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei) {
    int i_loc1 = source(*ei, g);
    int i_loc2 = target(*ei, g);

    const rt::Location &loc1 = int_loc_mapping[i_loc1];
    const rt::Location &loc2 = int_loc_mapping[i_loc2];

    double w = weight[*ei];
    double mi = -w;

    clt_edges.emplace_back(loc1, loc2, mi);
  }

  return clt_edges;
}

void ChowLuiTree::ConstructTreePrim(const JointModel &jm, const boost::optional<const rt::DenseOccGrid&> &dog) {
  double max_mi = log(2);

  library::timer::Timer t;
  t.Start();

  // Adapted from boost example

  // Typedef's for convience
  typedef boost::adjacency_list<boost::vecS, boost::vecS,
          boost::undirectedS, boost::no_property,
          boost::property<boost::edge_weight_t, double> > Graph;
  typedef boost::graph_traits<Graph>::vertex_descriptor VertexDescriptor;
  typedef std::pair<int, int> BoostEdge;

  // Making graph for boost;
  std::vector<BoostEdge> edges;
  std::vector<double> weights;

  // Mapping from node id (int) to location
  std::map<rt::Location, int> loc_int_mapping;
  std::vector<rt::Location> int_loc_mapping;

  // Get all edges
  int min_ij = -jm.GetNXY() / 2;
  int max_ij = min_ij + jm.GetNXY();

  int min_k = -jm.GetNZ() / 2;
  int max_k = min_k + jm.GetNZ();

  for (int i1=min_ij; i1 < max_ij; i1++) {
    for (int j1=min_ij; j1 < max_ij; j1++) {
      for (int k1=min_k; k1 < max_k; k1++) {
        rt::Location loc1(i1, j1, k1);

        if (dog && !dog->IsKnown(loc1)) {
          continue;
        }

        if (loc_int_mapping.count(loc1) == 0) {
          loc_int_mapping[loc1] = int_loc_mapping.size();
          int_loc_mapping.push_back(loc1);
        }

        int i_loc1 = loc_int_mapping[loc1];

        if (!jm.InRange(loc1)) {
          //printf("Out of range! %d, %d, %d\n",
          //    loc1.i, loc1.j, loc1.k);
          continue;
        }

        for (int i2 = min_ij; i2 < max_ij; i2++) {
          for (int j2 = min_ij; j2 < max_ij; j2++) {
            for (int k2 = min_k; k2 < max_k; k2++) {
              double di = i1 - i2;
              double dj = j1 - j2;
              double dk = k1 - k2;
              double d2 = di*di + dj*dj + dk*dk;
              if (sqrt(d2) > kMaxDistanceBetweenNodes_) {
                continue;
              }

              rt::Location loc2(i2, j2, k2);

              if (loc1 == loc2 || loc2 < loc1) {
                continue;
              }

              if (dog && !dog->IsKnown(loc2)) {
                continue;
              }

              if (jm.GetNumObservations(loc1, loc2) < kMinObservations_) {
                continue;
              }

              if (loc_int_mapping.count(loc2) == 0) {
                loc_int_mapping[loc2] = int_loc_mapping.size();
                int_loc_mapping.push_back(loc2);
              }

              int i_loc2 = loc_int_mapping[loc2];

              double mi = jm.ComputeMutualInformation(loc1, loc2);
              if (mi < kMinMutualInformation_) {
                continue;
              }

              double weight = max_mi - mi; // because we have minimum spanning tree but want max

              edges.push_back(BoostEdge(i_loc1, i_loc2));
              weights.push_back(weight);
            }
          }
        }
      }
    }
  }
  size_t num_edges = edges.size();
  size_t num_nodes = int_loc_mapping.size();
  //printf("Took %5.3f seconds to get %ld edges, %ld nodes\n", t.GetSeconds(), num_edges, num_nodes);

  Graph g(edges.begin(), edges.begin() + num_edges, weights.begin(), num_nodes);

  std::vector<VertexDescriptor> p(num_nodes);

  t.Start();
  boost::prim_minimum_spanning_tree(g, &p[0]);
  //printf("prim done in %5.3f sec\n", t.GetSeconds());

  double sum_mi = 0;

  for (size_t i = 0; i < p.size(); i++) {
    int int_my_loc = i;
    int int_parent_loc = p[i];

    if (int_my_loc == int_parent_loc) {
      // Is root node
      const auto &my_loc = int_loc_mapping[int_my_loc];

      Node node(my_loc, jm);
      nodes_.insert({my_loc, node});

      root_locs_.push_back(my_loc);
    } else {
      // Is child node
      const auto &my_loc = int_loc_mapping[int_my_loc];
      const auto &parent_loc = int_loc_mapping[int_parent_loc];

      Node node(my_loc, parent_loc, jm);
      nodes_.insert({my_loc, node});

      sum_mi += jm.ComputeMutualInformation(parent_loc, my_loc);
    }
  }

  //printf("Have %ld nodes, %ld roots in tree, %5.3f sum MI\n", nodes_.size(), root_locs_.size(), sum_mi);

  // Add children to parents
  for (const auto &it : nodes_) {
    const auto &node = it.second;
    if (node.HasParent()) {
      const auto &parent_loc = node.GetParentLocation();

      const auto &it_parent = nodes_.find(parent_loc);
      BOOST_ASSERT(it_parent != nodes_.end());

      it_parent->second.AddChild(node);
    }
  }
}

double ChowLuiTree::GetResolution() const {
  return resolution_;
}

const std::vector<rt::Location>& ChowLuiTree::GetRootLocs() const {
  return root_locs_;
}

const ChowLuiTree::Node& ChowLuiTree::GetNode(const rt::Location &loc) const {
  auto it = nodes_.find(loc);
  BOOST_ASSERT(it != nodes_.end());

  return it->second;
}

rt::OccGrid ChowLuiTree::Sample() const {
  // Random sampling
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine rand_engine(seed);

  // Occ grid map
  std::map<rt::Location, bool> sample_og;

  // Start from roots and traverse tree
  for (const auto &p : root_locs_) {
    auto it = nodes_.find(p);
    BOOST_ASSERT(it != nodes_.end());
    SampleHelper(it->second, &sample_og, &rand_engine);
  }

  // Spoof an occ grid
  std::vector<rt::Location> locs;
  std::vector<float> los;

  for (auto it : sample_og) {
    locs.push_back(it.first);
    los.push_back(it.second ? 1.0:-1.0);
  }

  rt::OccGrid og(locs, los, GetResolution());

  return og;
}

void ChowLuiTree::SampleHelper(const ChowLuiTree::Node &node_at, std::map<rt::Location, bool> *sample_og_pointer, std::default_random_engine *rand_engine) const {
  std::uniform_real_distribution<double> rand_unif(0.0, 1.0);

  auto &sample_og = *sample_og_pointer;

  // Assign this node
  if (node_at.HasParent()) {
    bool parent_occu = sample_og[node_at.GetParentLocation()];
    sample_og[node_at.GetLocation()] = rand_unif(*rand_engine) < node_at.GetConditionalProbability(parent_occu, true);
  } else {
    sample_og[node_at.GetLocation()] = rand_unif(*rand_engine) < node_at.GetMarginalProbability(true);
  }

  // Process children
  for (const auto &loc : node_at.GetChildren()) {
    auto it = nodes_.find(loc);
    BOOST_ASSERT(it != nodes_.end());
    SampleHelper(it->second, sample_og_pointer, rand_engine);
  }
}

double ChowLuiTree::EvaluateLogProbability(const rt::DenseOccGrid &dog, const EvalType &type) const {
  rt::DenseOccGrid my_dog(dog);

  double log_prob = 0;

  // Traverse through the tree
  for (const auto &root_loc : root_locs_) {
    const auto it_root = nodes_.find(root_loc);
    BOOST_ASSERT(it_root != nodes_.end());

    const auto &root_node = it_root->second;
    log_prob += EvaluateLogProbabilityHelper(root_node, &my_dog, type);
  }

  return log_prob;
}

double ChowLuiTree::EvaluateLogProbabilityHelper(const Node &node, rt::DenseOccGrid *dog, const EvalType &type) const {
  double log_prob = 0.0;

  const auto &loc = node.GetLocation();

  if (dog->IsKnown(loc)) {
    // Evaluate this update
    log_prob += GetNodeUpdate(node, dog, type);
  } else {
    // Put best guess (according to EvalType) of this location in the map so children can use it
    dog->Set(loc, GetNodeFillin(node, dog, type));
  }

  // Evaluate children
  for (const auto &child_loc : node.GetChildren()) {
    const auto it_child = nodes_.find(child_loc);
    BOOST_ASSERT(it_child != nodes_.end());

    const auto &child_node = it_child->second;
    log_prob += EvaluateLogProbabilityHelper(child_node, dog, type);
  }

  return log_prob;
}

double ChowLuiTree::GetNodeUpdate(const Node &node, rt::DenseOccGrid *dog, const EvalType &type) const {
  const auto &loc = node.GetLocation();
  bool obs = dog->GetProbability(loc) > 0.5;
  double p_obs = 0.0;

  if (node.HasParent()) {
    const auto &parent_loc = node.GetParentLocation();
    BOOST_ASSERT(dog->IsKnown(parent_loc));

    if (type == DENSE || type == APPROX_MARGINAL || type == APPROX_CONDITIONAL || type == APPROX_GREEDY) {
      // Get estimate of parent
      double p_parent_occu = dog->GetProbability(parent_loc);
      double p_parent_free = 1 - p_parent_occu;

      // Get conditionals
      double p_obs_parent_occu = node.GetConditionalProbability(true, obs);
      double p_obs_parent_free = node.GetConditionalProbability(false, obs);

      p_obs = p_obs_parent_occu * p_parent_occu + p_obs_parent_free * p_parent_free;
    } else if (type == MARGINAL) {
      p_obs = node.GetMarginalProbability(obs);
    } else {
      BOOST_ASSERT(false);
    }

  } else {
    p_obs = node.GetMarginalProbability(obs);
  }

  return log(p_obs);
}

double ChowLuiTree::GetNodeFillin(const Node &node, rt::DenseOccGrid *dog, const EvalType &type) const {
  double p_obs = 0.0;

  if (type == DENSE) {
    BOOST_ASSERT(false);      // Should never happen
  } else if (type == APPROX_MARGINAL || type == MARGINAL) {
    p_obs = node.GetMarginalProbability(true);
  } else if (type == APPROX_CONDITIONAL || type == APPROX_GREEDY) {
    if (node.HasParent()) {
      const auto &parent_loc = node.GetParentLocation();

      // Get estimate of parent
      double p_parent_occu = dog->GetProbability(parent_loc);
      double p_parent_free = 1 - p_parent_occu;

      // Get conditionals
      double p_obs_parent_occu = node.GetConditionalProbability(true, true);
      double p_obs_parent_free = node.GetConditionalProbability(false, true);

      p_obs = p_obs_parent_occu * p_parent_occu + p_obs_parent_free * p_parent_free;
    } else {
      p_obs = node.GetMarginalProbability(true);
    }

    if (type == APPROX_GREEDY) {
      p_obs = (p_obs < 0.5) ? 0.0:1.0;    // Clamp to 0 or 1
    }
  } else {
    BOOST_ASSERT(false);
  }

  return p_obs;
}

size_t ChowLuiTree::Size() const {
  return nodes_.size();
}

void ChowLuiTree::Save(const char *fn) const {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << (*this);
}

ChowLuiTree ChowLuiTree::Load(const char *fn) {
  ChowLuiTree t;
  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> t;

  return t;
}

} // namespace sim_world_occ_grids
} // namespace app
