#include "search-interface.h"
#include "search-strategies.h"
#include <cstdint>
#include <deque>
#include <functional>
#include <set>
#include <unordered_set>
#include <iostream>
#include <optional>
#include <algorithm>

// Wrapper over game state to help keep track of relationship between parent state and child state.
// Used for searching back through the search space when final state is found.
struct SearchNode {
	// Unique identifier of this search state
	uint64_t id;
	// Identifier of the parent, optional because initial state has no parent
	std::optional<uint64_t> prev;
	// Wrapped state
	SearchState state;
	// Action in the previous state that "got us here", optional because initial state wasn't "born" from an action
	std::optional<SearchAction> action;
	
	bool friend operator==(const SearchNode &a, const SearchNode &b) {
		return a.state == b.state;
	}
};

// Custom hasher for SearchNode which only hashes the internal state of SearchNode ignoring the bookkeeping.
struct SearchNodeHasher {
    std::size_t operator()(const SearchNode& key) const {
    	// DANGER: Someone decided to call this friend function 'hash', which clashes with std::hash. So you mustn't use 'using namespace std' otherwise compiler will spew absolutely horrifying error messages!
        return hash(key.state);
    }
};

// Hashes buffer buf of length len using the FNV-1 hashing algorithm (https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function)
size_t fnv_hash(const uint8_t* buf, size_t len) {
	constexpr uint64_t offset_basis = 0xcbf29ce484222325;
	constexpr uint64_t prime = 0x00000100000001b3;

	uint64_t hash = offset_basis;

	for(size_t i = 0; i < len; i++) {
		hash *= prime;
		hash ^= buf[i];
	}

	return hash;
}

size_t hash(const SearchState &state) {
	// We do not have access to the underlying GameState, so we just interpret the object
	// as bytes and hash those.
	const uint8_t* internal_state = reinterpret_cast<const uint8_t* >(&state.state_);
	return fnv_hash(internal_state, sizeof(state.state_));
}


bool operator==(const SearchState &a, const SearchState &b) {
	return a.state_ == b.state_;
}

std::vector<SearchAction> construct_solution(
	SearchNode& final_state,
	std::unordered_set<SearchNode, SearchNodeHasher>& explored
) {
	SearchNode* current = &final_state;
	std::vector<SearchAction> actions;

	// Traverse the backtree to initial state, collect actions along the way
	while(current->prev.has_value()) {
		actions.push_back(current->action.value()); // SAFETY: Action has value <=> action has previous, we checked previous, this is safe

		// Now we have to find the parent state
		for(auto node : explored) {
			if(node.id == current->prev) {
				current = &node;
			}
		}
	}

	// Actions are reversed from traversal, we need it it in the right order (from initial to final)
	reverse(actions.begin(), actions.end());


	return actions;
}

std::vector<SearchAction> BreadthFirstSearch::solve(const SearchState &init_state) {
	// NOTE: Tráví to brutálně moc času hledáním v EXPLORED
	// - Měl bych použít hešování
	// - Zjistit, jestli se položky v hešovací tabulce hýbají (a ukazatele na ně se tudíž nedají použít) - ANO, hýbou se, když se při vkládání překročí určitá mez, nelze použít ukazatele

	std::deque<SearchNode> frontier;
	std::unordered_set<SearchNode, SearchNodeHasher> explored;

	// ID is assigned based on order of processing, we increment it on every assignment using
	// the postfix ++ operator.
	uint64_t id = 0;

	frontier.push_front(SearchNode {id++, std::nullopt, init_state, std::nullopt});

	while(true) {
		if(frontier.empty()) {
			std::cout << "BFS found empty FRONTIER, this should never happen";
			return {};
		} else if (id > 1000) {
			std::cout << "Too long :(";
			return {};
		}

		SearchNode& work_node = frontier.front(); // SAFETY: We checked for emptyness, this is safe
		SearchState& work_state = work_node.state;

		if(explored.find(work_node) != explored.end()) {
			// If we already found this state, we skip it
			continue;
		} else if(work_state.isFinal()) {
			// If node is final, construct the solution and terminate
			// We are looking for states with specific IDs in explored set.
			return construct_solution(work_node, explored);
		} else {
			// Unpack working state, generate new states, add them to the frontier
			auto actions = work_state.actions();
			for(SearchAction action : actions) {
				SearchState new_state = action.execute(work_state);
				frontier.push_back(SearchNode {id++, work_node.id, new_state, action});
			}

			// Now we move the node from frontier to explored
			frontier.pop_front(); // SAFETY: front is work_node, so there is a front, popping is safe.
			explored.insert(work_node);
		}
	}

	return {};
}

std::vector<SearchAction> DepthFirstSearch::solve(const SearchState &init_state) {
	return {};
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const {
    return 0;
}

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state) {
	return {};
}
