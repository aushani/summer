#include "app/sim_world_occ_grids/chow_lui_tree.h"

#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

#include "library/timer/timer.h"

namespace app {
namespace sim_world_occ_grids {

ChowLuiTree::ChowLuiTree() {
}

ChowLuiTree::ChowLuiTree(const JointModel &jm) : resolution_(jm.GetResolution()) {
  boost::optional<const rt::DenseOccGrid&> dog;

  // Make edges
  auto edges = ConstructEdges(jm, dog);

  // Make tree
  MakeTree(edges, jm);
}

ChowLuiTree::ChowLuiTree(const JointModel &jm, const rt::DenseOccGrid &dog) : resolution_(jm.GetResolution()) {
  // Make edges
  auto edges = ConstructEdges(jm, dog);

  // Make tree
  MakeTree(edges, jm);
}

void ChowLuiTree::MakeTree(const std::vector<ChowLuiTree::Edge> &e, const JointModel &jm) {
  // Get tree edges
  std::vector<Edge> edges(e);

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
      parent_locs_.push_back(n.GetLocation());

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
    edges.erase(it);

    // TODO what about nodes with no edges?
  }
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

double ChowLuiTree::GetResolution() const {
  return resolution_;
}

const std::vector<rt::Location>& ChowLuiTree::GetParentLocs() const {
  return parent_locs_;
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

  // Start from parents and traverse tree
  for (const auto &p : parent_locs_) {
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

double ChowLuiTree::EvaluateLogProbability(const rt::DenseOccGrid &dog) const {
  double log_prob = 0;

  for (const auto &kv : nodes_) {
    const auto &loc = kv.first;
    const auto &node = kv.second;
    BOOST_ASSERT(dog.IsKnown(loc));

    bool obs = dog.GetProbability(loc) > 0.5;
    double p_obs = 0.0;

    if (!node.HasParent()) {
      p_obs = node.GetMarginalProbability(obs);
    } else {
      bool parent_obs = dog.GetProbability(node.GetParentLocation()) > 0.5;
      p_obs = node.GetConditionalProbability(parent_obs, obs);
    }

    log_prob += log(p_obs);
  }

  return log_prob;
}

double ChowLuiTree::EvaluateMarginalLogProbability(const rt::DenseOccGrid &dog) const {
  double log_prob = 0;

  for (const auto &kv : nodes_) {
    const auto &loc = kv.first;
    const auto &node = kv.second;

    if (!dog.IsKnown(loc)) {
      continue;
    }

    bool obs = dog.GetProbability(loc) > 0.5;
    double p_obs = node.GetMarginalProbability(obs);

    log_prob += log(p_obs);
  }

  return log_prob;
}

double ChowLuiTree::EvaluateApproxLogProbability(const rt::DenseOccGrid &dog) const {
  double log_prob = 0;

  for (const auto &kv : nodes_) {
    const auto &loc = kv.first;
    const auto &node = kv.second;

    if (!dog.IsKnown(loc)) {
      continue;
    }

    bool obs = dog.GetProbability(loc) > 0.5;
    double p_obs = 0.0;

    if (!node.HasParent()) {
      p_obs = node.GetMarginalProbability(obs);
    } else {
      const auto &parent_loc = node.GetParentLocation();

      if (dog.IsKnown(parent_loc)) {
        bool parent_obs = dog.GetProbability(node.GetParentLocation()) > 0.5;
        p_obs = node.GetConditionalProbability(parent_obs, obs);
      } else {
        double p_obs_occu = node.GetConditionalProbability(true, obs);
        double p_obs_free = node.GetConditionalProbability(false, obs);

        const auto it_parent = nodes_.find(parent_loc);
        BOOST_ASSERT(it_parent != nodes_.end());

        double p_parent_occu = it_parent->second.GetMarginalProbability(true);
        double p_parent_free = it_parent->second.GetMarginalProbability(false);

        p_obs = p_obs_occu * p_parent_occu + p_obs_free * p_parent_free;
      }
    }

    log_prob += log(p_obs);
  }

  return log_prob;
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