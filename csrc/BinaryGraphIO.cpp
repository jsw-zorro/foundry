// Binary CUDA Graph I/O — write and read .cugraph format
// See include/BinaryGraphFormat.h for format specification.

#include "CUDAGraph.h"
#include "BinaryGraphFormat.h"
#include <boost/json.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace foundry {

void commit_file(const std::string& tmp_path, const std::string& path, bool written) {
  if (written && std::rename(tmp_path.c_str(), path.c_str()) == 0)
    return;
  const int err = errno;
  std::remove(tmp_path.c_str());
  TORCH_CHECK(false, "Failed to write ", path, ": ",
              written ? std::strerror(err) : "the write did not complete");
}

void CUDAGraph::save_binary(const std::string& bin_path, const boost::json::object& root) {
  namespace json = boost::json;
  namespace bf = binary_format;

  const json::array& nodes_array = root.at("nodes").as_array();
  const json::array& deps_array =
      root.contains("dependencies") ? root.at("dependencies").as_array() : json::array{};
  const json::array& generators_array = root.at("generators").as_array();

  // ---- Build string table (deduplicated function names) ----
  std::vector<char> string_table;
  std::unordered_map<std::string, uint32_t> string_offsets;
  auto intern_string = [&](const std::string& s) -> uint32_t {
    auto it = string_offsets.find(s);
    if (it != string_offsets.end())
      return it->second;
    uint32_t offset = static_cast<uint32_t>(string_table.size());
    string_table.insert(string_table.end(), s.begin(), s.end());
    string_table.push_back('\0');
    string_offsets[s] = offset;
    return offset;
  };

  std::string topology_key =
      root.contains("topology_key") ? std::string(root.at("topology_key").as_string()) : "";

  // ---- Build node entries and param data ----
  std::vector<bf::BinNodeEntry> node_entries;
  std::vector<bf::BinParamEntry> param_index;
  std::vector<uint8_t> param_data;
  std::vector<uint8_t> arg_buffer_data;
  node_entries.reserve(nodes_array.size());

  // Helper to decode hex string to binary
  auto hex_decode = [](const std::string& hex, std::vector<uint8_t>& out) {
    for (size_t j = 0; j < hex.size(); j += 2) {
      uint8_t byte = static_cast<uint8_t>(std::stoul(hex.substr(j, 2), nullptr, 16));
      out.push_back(byte);
    }
  };

  // Helper to parse kernel_node_attrs flags from JSON
  auto parse_kna_flags = [](const json::object& kna, bf::BinKernelNode& k) {
    k.kna_flags = 0;
    if (kna.contains("attrQueryAvailable")) {
      k.kna_flags |= bf::KNA_ATTR_QUERY_AVAILABLE;
      k.kna_attrQueryAvailable = kna.at("attrQueryAvailable").as_bool() ? 1 : 0;
    }
    if (kna.contains("clusterDimX")) {
      k.kna_flags |= bf::KNA_CLUSTER_DIM;
      k.kna_clusterDimX = kna.at("clusterDimX").to_number<uint32_t>();
      k.kna_clusterDimY = kna.at("clusterDimY").to_number<uint32_t>();
      k.kna_clusterDimZ = kna.at("clusterDimZ").to_number<uint32_t>();
    }
    if (kna.contains("preferredClusterDimX")) {
      k.kna_flags |= bf::KNA_PREFERRED_CLUSTER_DIM;
      k.kna_preferredClusterDimX = kna.at("preferredClusterDimX").to_number<uint32_t>();
      k.kna_preferredClusterDimY = kna.at("preferredClusterDimY").to_number<uint32_t>();
      k.kna_preferredClusterDimZ = kna.at("preferredClusterDimZ").to_number<uint32_t>();
    }
    if (kna.contains("clusterSchedulingPolicyPreference")) {
      k.kna_flags |= bf::KNA_CLUSTER_SCHEDULING;
      k.kna_clusterSchedulingPolicy =
          kna.at("clusterSchedulingPolicyPreference").to_number<int32_t>();
    }
    if (kna.contains("cooperative")) {
      k.kna_flags |= bf::KNA_COOPERATIVE;
      k.kna_cooperative = kna.at("cooperative").to_number<int32_t>();
    }
    if (kna.contains("priority")) {
      k.kna_flags |= bf::KNA_PRIORITY;
      k.kna_priority = kna.at("priority").to_number<int32_t>();
    }
    if (kna.contains("memSyncDomain")) {
      k.kna_flags |= bf::KNA_MEM_SYNC_DOMAIN;
      k.kna_memSyncDomain = kna.at("memSyncDomain").to_number<int32_t>();
    }
    if (kna.contains("memSyncDomainMapDefault")) {
      k.kna_flags |= bf::KNA_MEM_SYNC_DOMAIN_MAP;
      k.kna_memSyncDomainMapDefault =
          static_cast<uint8_t>(kna.at("memSyncDomainMapDefault").to_number<int>());
      k.kna_memSyncDomainMapRemote =
          static_cast<uint8_t>(kna.at("memSyncDomainMapRemote").to_number<int>());
    }
    if (kna.contains("preferredSharedMemCarveout")) {
      k.kna_flags |= bf::KNA_SHARED_MEM_CARVEOUT;
      k.kna_preferredSharedMemCarveout = kna.at("preferredSharedMemCarveout").to_number<uint32_t>();
    }
  };

  // Attributes the node table cannot hold: an access-policy window anywhere,
  // or programmatic stream serialization / device-updatable in the common
  // attributes. Any of them leaves FLAG_COMPLETE_KERNEL_ATTRS unset, so LOAD
  // builds this graph's template from the JSON.
  bool complete_kernel_attrs = true;
  auto has_unrepresentable = [](const json::object& kna, bool common) {
    return kna.contains("accessPolicyWindowNumBytes") ||
           (common &&
            (kna.contains("programmaticStreamSerialization") || kna.contains("deviceUpdatable")));
  };

  // Common kernel attrs
  bool has_common_attrs = root.contains("common_kernel_node_attrs");
  bf::BinCommonKernelAttrs common_attrs = {};
  if (has_common_attrs) {
    const json::object& ca = root.at("common_kernel_node_attrs").as_object();
    if (has_unrepresentable(ca, true))
      complete_kernel_attrs = false;
    // Reuse the kna parser with a temporary BinKernelNode to extract flags
    bf::BinKernelNode tmp = {};
    parse_kna_flags(ca, tmp);
    common_attrs.flags = tmp.kna_flags;
    common_attrs.clusterDimX = tmp.kna_clusterDimX;
    common_attrs.clusterDimY = tmp.kna_clusterDimY;
    common_attrs.clusterDimZ = tmp.kna_clusterDimZ;
    common_attrs.preferredClusterDimX = tmp.kna_preferredClusterDimX;
    common_attrs.preferredClusterDimY = tmp.kna_preferredClusterDimY;
    common_attrs.preferredClusterDimZ = tmp.kna_preferredClusterDimZ;
    common_attrs.clusterSchedulingPolicy = tmp.kna_clusterSchedulingPolicy;
    common_attrs.cooperative = tmp.kna_cooperative;
    common_attrs.priority = tmp.kna_priority;
    common_attrs.memSyncDomain = tmp.kna_memSyncDomain;
    common_attrs.memSyncDomainMapDefault = tmp.kna_memSyncDomainMapDefault;
    common_attrs.memSyncDomainMapRemote = tmp.kna_memSyncDomainMapRemote;
    common_attrs.attrQueryAvailable = tmp.kna_attrQueryAvailable;
    common_attrs.preferredSharedMemCarveout = tmp.kna_preferredSharedMemCarveout;
  }

  // ---- Build node entries ----
  for (const auto& node_val : nodes_array) {
    const json::object& no = node_val.as_object();
    bf::BinNodeEntry entry = {};
    entry.node_id = no.at("id").to_number<uint32_t>();

    std::string type_str(no.at("type").as_string());

    if (type_str == "KernelNode") {
      entry.type = bf::NODE_KERNEL;
      const json::object& p = no.at("params").as_object();
      auto& k = entry.kernel;

      k.blockDimX = p.at("blockDimX").to_number<uint32_t>();
      k.blockDimY = p.at("blockDimY").to_number<uint32_t>();
      k.blockDimZ = p.at("blockDimZ").to_number<uint32_t>();
      k.gridDimX = p.at("gridDimX").to_number<uint32_t>();
      k.gridDimY = p.at("gridDimY").to_number<uint32_t>();
      k.gridDimZ = p.at("gridDimZ").to_number<uint32_t>();
      k.sharedMemBytes = p.at("sharedMemBytes").to_number<uint32_t>();

      std::string fn(p.at("function_name").as_string());
      k.function_name_offset = intern_string(fn);
      k.function_name_length = static_cast<uint32_t>(fn.size());
      k.binary_hash = p.at("kernel_source_binary_hash").to_number<uint64_t>();

      // func_attrs
      if (p.contains("func_attrs")) {
        const json::object& fa = p.at("func_attrs").as_object();
        k.max_dynamic_shared_size_bytes =
            fa.at("max_dynamic_shared_size_bytes").to_number<int32_t>();
        k.preferred_shared_memory_carveout =
            fa.at("preferred_shared_memory_carveout").to_number<int32_t>();
        k.cluster_scheduling_policy_preference =
            fa.contains("cluster_scheduling_policy_preference")
                ? fa.at("cluster_scheduling_policy_preference").to_number<int32_t>()
                : 0;
        k.required_cluster_width = fa.contains("required_cluster_width")
                                       ? fa.at("required_cluster_width").to_number<int32_t>()
                                       : 0;
        k.required_cluster_height = fa.contains("required_cluster_height")
                                        ? fa.at("required_cluster_height").to_number<int32_t>()
                                        : 0;
        k.required_cluster_depth = fa.contains("required_cluster_depth")
                                       ? fa.at("required_cluster_depth").to_number<int32_t>()
                                       : 0;
      }

      // kernel_node_attrs (per-node overrides)
      if (p.contains("kernel_node_attrs")) {
        const json::object& kna = p.at("kernel_node_attrs").as_object();
        parse_kna_flags(kna, k);
        if (auto* v = kna.if_contains("programmaticStreamSerialization")) {
          k.ext_flags |= bf::KNX_PROGRAMMATIC_STREAM_SERIALIZATION;
          k.ext_programmatic_stream_serialization = static_cast<uint8_t>(v->to_number<int>());
        }
        if (auto* v = kna.if_contains("deviceUpdatable")) {
          k.ext_flags |= bf::KNX_DEVICE_UPDATABLE;
          k.ext_device_updatable = static_cast<uint8_t>(v->to_number<int>());
        }
        if (has_unrepresentable(kna, false))
          complete_kernel_attrs = false;
      }

      // Kernel params
      k.param_index_offset = static_cast<uint32_t>(param_index.size() * sizeof(bf::BinParamEntry));
      const json::array& kp_array = p.at("kernelParams").as_array();
      k.num_params = static_cast<uint32_t>(kp_array.size());

      for (const auto& kp_val : kp_array) {
        const json::object& kp = kp_val.as_object();
        bf::BinParamEntry pe = {};

        if (kp.contains("value_hex")) {
          const std::string hex_str(kp.at("value_hex").as_string());
          pe.data_offset = static_cast<uint32_t>(param_data.size());
          pe.size = static_cast<uint32_t>(hex_str.size() / 2);
          hex_decode(hex_str, param_data);
        } else {
          // Metadata-only (extra-style) — offset/size descriptors
          pe.data_offset = kp.at("offset").to_number<uint32_t>();
          pe.size = kp.at("size").to_number<uint32_t>();
        }
        param_index.push_back(pe);
      }

      // Extra arg buffer
      const std::string extra_hex(p.contains("extra_argBuffer_hex")
                                      ? std::string(p.at("extra_argBuffer_hex").as_string())
                                      : "");
      if (!extra_hex.empty()) {
        k.arg_buffer_offset = static_cast<uint32_t>(arg_buffer_data.size());
        k.arg_buffer_size = static_cast<uint32_t>(extra_hex.size() / 2);
        hex_decode(extra_hex, arg_buffer_data);
      } else {
        k.arg_buffer_offset = 0xFFFFFFFF;
        k.arg_buffer_size = 0;
      }

    } else if (type_str == "MemcpyNode") {
      entry.type = bf::NODE_MEMCPY;
      const json::object& p = no.at("params").as_object();
      auto& m = entry.memcpy;
      m.Depth = p.at("Depth").to_number<uint64_t>();
      m.Height = p.at("Height").to_number<uint64_t>();
      m.WidthInBytes = p.at("WidthInBytes").to_number<uint64_t>();
      m.dstDevice = p.at("dstDevice").to_number<uint64_t>();
      m.dstHeight = p.at("dstHeight").to_number<uint64_t>();
      m.dstLOD = p.at("dstLOD").to_number<uint64_t>();
      m.dstMemoryType = p.at("dstMemoryType").to_number<int32_t>();
      m.dstPitch = p.at("dstPitch").to_number<uint64_t>();
      m.dstXInBytes = p.at("dstXInBytes").to_number<uint64_t>();
      m.dstY = p.at("dstY").to_number<uint64_t>();
      m.dstZ = p.at("dstZ").to_number<uint64_t>();
      m.srcDevice = p.at("srcDevice").to_number<uint64_t>();
      m.srcHeight = p.at("srcHeight").to_number<uint64_t>();
      m.srcLOD = p.at("srcLOD").to_number<uint64_t>();
      m.srcMemoryType = p.at("srcMemoryType").to_number<int32_t>();
      m.srcPitch = p.at("srcPitch").to_number<uint64_t>();
      m.srcXInBytes = p.at("srcXInBytes").to_number<uint64_t>();
      m.srcY = p.at("srcY").to_number<uint64_t>();
      m.srcZ = p.at("srcZ").to_number<uint64_t>();

    } else if (type_str == "MemsetNode") {
      entry.type = bf::NODE_MEMSET;
      const json::object& p = no.at("params").as_object();
      auto& m = entry.memset;
      m.dst = p.at("dst").to_number<uint64_t>();
      m.elementSize = p.at("elementSize").to_number<uint32_t>();
      m.height = p.at("height").to_number<uint64_t>();
      m.pitch = p.at("pitch").to_number<uint64_t>();
      m.value = p.at("value").to_number<uint32_t>();
      m.width = p.at("width").to_number<uint64_t>();

    } else if (type_str == "EventRecordNode") {
      entry.type = bf::NODE_EVENT_RECORD;
      entry.event.event_id = no.at("params").as_object().at("event_id").to_number<int32_t>();

    } else if (type_str == "EventWaitNode") {
      entry.type = bf::NODE_EVENT_WAIT;
      entry.event.event_id = no.at("params").as_object().at("event_id").to_number<int32_t>();

    } else if (type_str == "EmptyNode") {
      entry.type = bf::NODE_EMPTY;
    }

    node_entries.push_back(entry);
  }

  // ---- Dependencies ----
  std::vector<bf::BinDependency> deps;
  deps.reserve(deps_array.size());
  for (const auto& dep_val : deps_array) {
    const json::object& d = dep_val.as_object();
    bf::BinDependency bd{};
    bd.from_id = d.at("from").to_number<uint32_t>();
    bd.to_id = d.at("to").to_number<uint32_t>();
    // Edge data (PDL programmatic ports); absent fields = default edge.
    if (auto* v = d.if_contains("from_port")) {
      bd.from_port = (uint8_t)v->to_number<int>();
    }
    if (auto* v = d.if_contains("to_port")) {
      bd.to_port = (uint8_t)v->to_number<int>();
    }
    if (auto* v = d.if_contains("edge_type")) {
      bd.edge_type = (uint8_t)v->to_number<int>();
    }
    deps.push_back(bd);
  }

  // ---- Generators ----
  std::vector<bf::BinGenerator> gens;
  gens.reserve(generators_array.size());
  for (const auto& gen_val : generators_array) {
    const json::object& g = gen_val.as_object();
    auto ptr = [&](const char* key) {
      return g.contains(key) ? g.at(key).to_number<uint64_t>() : uint64_t{0};
    };
    gens.push_back({g.at("id").to_number<uint64_t>(), g.at("seed").to_number<uint64_t>(),
                    g.at("wholegraph_increment").to_number<uint64_t>(), ptr("seed_ptr"),
                    ptr("offset_ptr")});
  }

  // ---- JSON sections (allocator_events, output_tensors) ----
  std::string alloc_events_json =
      root.contains("allocator_events") ? json::serialize(root.at("allocator_events")) : "";
  std::string output_tensors_json =
      root.contains("output_tensors") ? json::serialize(root.at("output_tensors")) : "";

  // ---- Compute layout ----
  uint32_t num_sections = bf::SECTION_COUNT;
  uint64_t offset = sizeof(bf::FileHeader) + num_sections * sizeof(bf::SectionEntry);

  bf::SectionEntry sections[bf::SECTION_COUNT] = {};
  auto place_section = [&](bf::SectionType type, uint64_t size) {
    sections[type].section_type = type;
    sections[type].offset = offset;
    sections[type].size = size;
    offset += size;
    offset = (offset + 7) & ~7ULL;  // align to 8 bytes
  };

  place_section(bf::SECTION_STRING_TABLE, string_table.size());
  place_section(bf::SECTION_NODE_TABLE, node_entries.size() * sizeof(bf::BinNodeEntry));
  place_section(bf::SECTION_DEPENDENCY_TABLE, deps.size() * sizeof(bf::BinDependency));
  place_section(bf::SECTION_KERNEL_PARAM_INDEX, param_index.size() * sizeof(bf::BinParamEntry));
  place_section(bf::SECTION_KERNEL_PARAM_DATA, param_data.size());
  place_section(bf::SECTION_ARG_BUFFER_DATA, arg_buffer_data.size());
  place_section(bf::SECTION_COMMON_KERNEL_ATTRS,
                has_common_attrs ? sizeof(bf::BinCommonKernelAttrs) : 0);
  place_section(bf::SECTION_GENERATORS_TABLE, gens.size() * sizeof(bf::BinGenerator));
  place_section(bf::SECTION_ALLOCATOR_EVENTS, alloc_events_json.size());
  place_section(bf::SECTION_OUTPUT_TENSORS, output_tensors_json.size());
  place_section(bf::SECTION_TOPOLOGY_KEY, topology_key.size());

  // ---- Write ----
  // To a temporary name, renamed when complete: a failed write (a full disk,
  // EIO) raises instead of leaving a truncated file that LOAD would read.
  const std::string tmp_path = bin_path + ".tmp";
  std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
  TORCH_CHECK(out.is_open(), "Failed to open file for writing: ", tmp_path);

  bf::FileHeader header = {};
  memcpy(header.magic, bf::MAGIC, 8);
  header.version = bf::FORMAT_VERSION;
  header.flags = 0;
  if (has_common_attrs)
    header.flags |= bf::FLAG_HAS_COMMON_KERNEL_ATTRS;
  if (!deps.empty())
    header.flags |= bf::FLAG_HAS_DEPENDENCIES;
  if (!alloc_events_json.empty())
    header.flags |= bf::FLAG_HAS_ALLOCATOR_EVENTS;
  if (!output_tensors_json.empty())
    header.flags |= bf::FLAG_HAS_OUTPUT_TENSORS;
  if (!gens.empty())
    header.flags |= bf::FLAG_HAS_GENERATORS;
  if (complete_kernel_attrs)
    header.flags |= bf::FLAG_COMPLETE_KERNEL_ATTRS;
  header.num_nodes = static_cast<uint32_t>(node_entries.size());
  header.num_dependencies = static_cast<uint32_t>(deps.size());
  header.num_generators = static_cast<uint32_t>(gens.size());
  header.num_sections = num_sections;
  header.num_strings = static_cast<uint32_t>(string_offsets.size());

  out.write(reinterpret_cast<const char*>(&header), sizeof(header));
  out.write(reinterpret_cast<const char*>(sections), sizeof(sections));

  auto write_padded = [&](const void* data, size_t size) {
    if (size > 0)
      out.write(static_cast<const char*>(data), size);
    size_t pad = ((size + 7) & ~7ULL) - size;
    if (pad > 0) {
      static const char zeros[8] = {};
      out.write(zeros, pad);
    }
  };

  write_padded(string_table.data(), string_table.size());
  write_padded(node_entries.data(), node_entries.size() * sizeof(bf::BinNodeEntry));
  write_padded(deps.data(), deps.size() * sizeof(bf::BinDependency));
  write_padded(param_index.data(), param_index.size() * sizeof(bf::BinParamEntry));
  write_padded(param_data.data(), param_data.size());
  write_padded(arg_buffer_data.data(), arg_buffer_data.size());
  if (has_common_attrs)
    write_padded(&common_attrs, sizeof(common_attrs));
  write_padded(gens.data(), gens.size() * sizeof(bf::BinGenerator));
  write_padded(alloc_events_json.data(), alloc_events_json.size());
  write_padded(output_tensors_json.data(), output_tensors_json.size());
  write_padded(topology_key.data(), topology_key.size());

  out.close();
  commit_file(tmp_path, bin_path, !out.fail());
}

// ============================================================================
// Lightweight binary reader: just reads file + validates header.
// No JSON construction. Used for direct-access in prepare_on_demand_graph.
// ============================================================================

// Every count, offset and range that LOAD reads from a .cugraph, checked against
// the file. Returns the first problem found, or an empty string.
static std::string check_binary_graph_file(const BinaryGraphFile& f) {
  namespace bf = binary_format;
  const uint64_t file_size = f.data.size();
  const bf::FileHeader& h = f.header;
  if (h.num_sections < bf::SECTION_COUNT)
    return "the section directory has " + std::to_string(h.num_sections) + " entries";
  if (sizeof(bf::FileHeader) + uint64_t(h.num_sections) * sizeof(bf::SectionEntry) > file_size)
    return "the section directory extends past the end of the file";
  for (uint32_t t = 0; t < bf::SECTION_COUNT; ++t) {
    const bf::SectionEntry& s = f.sections[t];
    if (s.offset > file_size || s.size > file_size - s.offset)
      return "section " + std::to_string(t) + " extends past the end of the file";
  }
  auto in_range = [&](bf::SectionType t, uint64_t offset, uint64_t length) {
    const uint64_t size = f.section_size(t);
    return offset <= size && length <= size - offset;
  };
  const uint64_t dep_record =
      h.version >= 2 ? sizeof(bf::BinDependency) : sizeof(bf::BinDependencyV1);
  if (!in_range(bf::SECTION_NODE_TABLE, 0, uint64_t(h.num_nodes) * sizeof(bf::BinNodeEntry)))
    return "the node table is shorter than its " + std::to_string(h.num_nodes) + " nodes";
  if (!in_range(bf::SECTION_DEPENDENCY_TABLE, 0, uint64_t(h.num_dependencies) * dep_record))
    return "the dependency table is shorter than its " + std::to_string(h.num_dependencies) +
           " records";
  if (!in_range(bf::SECTION_GENERATORS_TABLE, 0,
                uint64_t(h.num_generators) * bf::generator_record_size(h.version)))
    return "the generator table is shorter than its " + std::to_string(h.num_generators) +
           " records";
  if ((h.flags & bf::FLAG_HAS_COMMON_KERNEL_ATTRS) &&
      !in_range(bf::SECTION_COMMON_KERNEL_ATTRS, 0, sizeof(bf::BinCommonKernelAttrs)))
    return "the common kernel attributes are truncated";

  std::unordered_set<uint32_t> ids;
  ids.reserve(h.num_nodes);
  const bf::BinNodeEntry* nodes = f.node_table();
  for (uint32_t i = 0; i < h.num_nodes; ++i) {
    const bf::BinNodeEntry& e = nodes[i];
    const std::string node = "node " + std::to_string(i);
    if (e.type > bf::NODE_EMPTY)
      return node + " has unknown type " + std::to_string(e.type);
    if (!ids.insert(e.node_id).second)
      return node + " repeats node id " + std::to_string(e.node_id);
    if (e.type != bf::NODE_KERNEL)
      continue;
    const bf::BinKernelNode& k = e.kernel;
    if (!in_range(bf::SECTION_STRING_TABLE, k.function_name_offset, k.function_name_length))
      return node + ": the function name is outside the string table";
    if (!in_range(bf::SECTION_KERNEL_PARAM_INDEX, k.param_index_offset,
                  uint64_t(k.num_params) * sizeof(bf::BinParamEntry)))
      return node + ": the parameter list is outside the parameter index";
    if (k.arg_buffer_offset != 0xFFFFFFFF) {
      // Extra-style launch: the parameters are offsets into this buffer.
      if (!in_range(bf::SECTION_ARG_BUFFER_DATA, k.arg_buffer_offset, k.arg_buffer_size))
        return node + ": the argument buffer is outside its section";
      continue;
    }
    const auto* params = reinterpret_cast<const bf::BinParamEntry*>(
        f.section_ptr<uint8_t>(bf::SECTION_KERNEL_PARAM_INDEX) + k.param_index_offset);
    for (uint32_t j = 0; j < k.num_params; ++j) {
      if (!in_range(bf::SECTION_KERNEL_PARAM_DATA, params[j].data_offset, params[j].size))
        return node + ": parameter " + std::to_string(j) + " is outside the parameter data";
    }
  }

  const uint8_t* deps = f.section_ptr<uint8_t>(bf::SECTION_DEPENDENCY_TABLE);
  for (uint32_t i = 0; i < h.num_dependencies; ++i) {
    // from_id and to_id lead both record versions.
    bf::BinDependencyV1 d;
    memcpy(&d, deps + i * dep_record, sizeof(d));
    if (!ids.count(d.from_id) || !ids.count(d.to_id))
      return "dependency " + std::to_string(i) + " names a node that is not in the node table";
  }
  return "";
}

BinaryGraphFile read_binary_graph_file(const std::string& bin_path) {
  namespace bf = binary_format;
  BinaryGraphFile result;

  std::ifstream in(bin_path, std::ios::binary | std::ios::ate);
  if (!in.is_open())
    return result;

  const std::streamoff end = in.tellg();
  if (end < static_cast<std::streamoff>(sizeof(bf::FileHeader)))
    return result;
  const size_t file_size = static_cast<size_t>(end);

  in.seekg(0);
  result.data.resize(file_size);
  in.read(reinterpret_cast<char*>(result.data.data()), file_size);
  if (static_cast<size_t>(in.gcount()) != file_size) {
    fprintf(stderr, "[foundry] WARNING: %s: read %lld of %zu bytes; LOAD uses the JSON\n",
            bin_path.c_str(), static_cast<long long>(in.gcount()), file_size);
    result.data.clear();
    return result;
  }
  in.close();

  memcpy(&result.header, result.data.data(), sizeof(bf::FileHeader));
  if (!bf::validate_header(result.header)) {
    result.data.clear();
    return result;
  }

  result.sections =
      reinterpret_cast<const bf::SectionEntry*>(result.data.data() + sizeof(bf::FileHeader));
  const std::string problem = check_binary_graph_file(result);
  if (!problem.empty()) {
    // A damaged binary (for example a copy cut short) is not used; the graph
    // is read from its JSON.
    fprintf(stderr, "[foundry] WARNING: %s: %s; LOAD uses the JSON\n", bin_path.c_str(),
            problem.c_str());
    result.data.clear();
    result.sections = nullptr;
  }
  return result;
}

}  // namespace foundry
