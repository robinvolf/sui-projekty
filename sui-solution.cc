#include "search-interface.h"
#include "search-strategies.h"
#include <deque>
#include <set>
#include <iostream>
#include <optional>
#include <algorithm>

using namespace std;

// Node of a tree, which only has references to the previous state.
// This is used for tracking the path from the final state *back* to the initial one.
// Field action stores the Action through which you get from parent state to current state.
//
// ### Optional
// The action and previous node pointer are optional (they are nonsensical for the initial state).
//
// ### Pointers
// In this data structure, the nodes are neved moved in memory, so pointers to the nodes
// are never invalidated.
//
// ### Cleanup 
// Requires manual cleanup unfortunately, use the function cleanup_backtree_nodes(). This deletes only those ones, which are provided using the iterator. This is C++, you're on your own.
struct BacktreeNode {
	SearchState state;
	optional<SearchAction> action;
	optional<BacktreeNode*> prev;
};

// Creates a new Backtree node with the provided state, action and pointer to parent node.
BacktreeNode* new_node(SearchState state,
                       optional<SearchAction> action,
                       optional<BacktreeNode*> prev)
{
	BacktreeNode* new_node = new BacktreeNode {state, action, prev};
	return new_node;
}

// Cleans up all backtree nodes from the iterator.
template<typename Iterator>
void cleanup_backtree_nodes(Iterator begin, Iterator end) {
    for (Iterator iter = begin; iter != end; ++iter) {
    	BacktreeNode* node = *iter;
    	delete node;
    }
}

// Custom comparison function for std::set
// We don't want to actions or pointers, only the state itself.
struct CompareBacktreeNodes {
    bool operator()(const BacktreeNode* lhs, const BacktreeNode* rhs) const {
		return lhs->state < rhs->state;
	}
};

vector<SearchAction> BreadthFirstSearch::solve(const SearchState &init_state) {
	// Plán:
	// - Co budu používat jako frontu pro FRONTIER?
	// - Co budu používat jako seznam pro EXPLORED
	// 
	// Agloritmus
	// 1. Vyber první stav z fronty
	// 2. Je finální?
	// 3. Pushni všechny stavy z něj rozbalené na konec fronty
	// 4. A furt

	// SAFETY: during cleanup, all the nodes are either in frontier or explored,
	// so by cleaning up those, we guarantee cleanup of everything.
	deque<BacktreeNode*> frontier;
	set<BacktreeNode*, CompareBacktreeNodes> explored;

	frontier.push_front(new_node(init_state, nullopt, nullopt));

	while(true) {
		if(frontier.empty()) {
			cout << "BFS found empty FRONTIER, this should never happen";
			return {};
		}

		BacktreeNode* work_node = frontier.front(); // SAFETY: We checked for emptyness, this is safe
		SearchState work_state = work_node->state;

		if(explored.find(work_node) != explored.end()) {
			// If we already found this state, we skip it
			continue;
		} else if(work_state.isFinal()) {
			// If node is final, construct the solution and terminate
			BacktreeNode* current = work_node;
			vector<SearchAction> actions;

			// Traverse the backtree to initial state, collect actions along the way
			while(current->prev.has_value()) {
				actions.push_back(current->action.value()); // SAFETY: Action has value <=> action has previous, we checked previous, this is safe
				current = current->prev.value(); // SAFETY: We checked in while condition, this is safe
			}

			// Actions are reversed from traversal, we need it it in the right order (from initial to final)
			reverse(actions.begin(), actions.end());


			// Cleanup of frontier and expanded
			cleanup_backtree_nodes(frontier.begin(), frontier.end());
			cleanup_backtree_nodes(explored.begin(), explored.end());
			
			return actions;
		} else {
			// Unpack working state, generate new states, add them to the frontier
			auto actions = work_state.actions();
			for(SearchAction action : actions) {
				SearchState new_state = action.execute(work_state);
				frontier.push_back(new_node(new_state, optional(action), optional(work_node)));
			}

			// Now we move the node from frontier to explored
			frontier.pop_front(); // SAFETY: front is work_node, so there is a front, popping is safe.
			explored.insert(work_node);
		}
	}

	// SAFETY: We should get here only if we failed to find the solution, if that happens
	// just break out of the loop, cleanup will be performed here.

	// Cleanup of frontier and expanded
	cleanup_backtree_nodes(frontier.begin(), frontier.end());
	cleanup_backtree_nodes(explored.begin(), explored.end());

	return {};
}

vector<SearchAction> DepthFirstSearch::solve(const SearchState &init_state) {
	return {};
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const {
    return 0;
}

vector<SearchAction> AStarSearch::solve(const SearchState &init_state) {
	return {};
}
